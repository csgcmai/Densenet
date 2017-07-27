'''
References:

Gao Huang, Zhuang Liu, Kilian Q. Weinberger, Laurens van der Maaten. "Densely Connected Convolutional Networks"
'''
from __future__ import division
#import find_mxnet
#assert find_mxnet
import mxnet as mx
import argparse
import os
import logging
import numpy as np

import pdb

parser = argparse.ArgumentParser(description='train an image classifer on cifar10')
parser.add_argument('--data-dir', type=str, default='cifar10/',
                    help='the input data directory')
parser.add_argument('--gpus', type=str,
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--num-examples', type=int, default=50000,
                    help='the number of training examples')
parser.add_argument('--batch-size', type=int, default=128,
                    help='the batch size')
parser.add_argument('--lr', type=float, default=0.1,
                    help='the initial learning rate')
parser.add_argument('--model-prefix', type=str,
                    help='the prefix of the model to load')
parser.add_argument('--save-model-prefix', type=str,
                    help='the prefix of the model to save')
parser.add_argument('--num-epochs', type=int, default=300,
                    help='the number of training epochs')
parser.add_argument('--load-epoch', type=int,
                    help='load the model on an epoch using the model-prefix')
parser.add_argument('--kv-store', type=str, default='local',
                    help='the kvstore type')
parser.add_argument('--log-file', type=str, default='log', help='the name of log file')
parser.add_argument('--log-dir', type=str, default='cifar10', help='directory of the log file')
args = parser.parse_args()

def GetTrainableParameters(symbol, executor):
    param_keys = symbol.list_arguments()
    arg = []
    grad = []
    for name in param_keys:
        if name not in ['data','label']:
            arg.append(executor.arg_dict[name])
            grad.append(executor.grad_dict[name])
    return arg, grad

def PrintShapes(symbol, executor):
    netwark_args, _ = GetTrainableParameters(symbol, executor)
    print 'Network shape checking:'
    total_params = 0
    for arg in netwark_args:
        shape = arg.shape
        print shape
        tmp_params = 1
        for dim in shape:
            tmp_params *= dim
        total_params += tmp_params
    return total_params

# Note that the random circular shift is implemented here
# The random seed is set by numpy
class IrisIter(mx.io.DataIter):
    def __init__(self, irisIter, maskIter, batch_size):
        super(IrisIter, self).__init__(batch_size)
        self.irisIter = irisIter
        self.maskIter = maskIter
        
        data_name = irisIter.provide_data[0][0]
        data_shape = irisIter.provide_data[0][1]
        data_desc = [mx.io.DataDesc(data_name, (data_shape[0],data_shape[1]*2)+data_shape[2:4])]

        assert irisIter.provide_data == maskIter.provide_data
        assert irisIter.provide_label == maskIter.provide_label

        self.width = data_shape[3]
        self.shift_max = data_shape[3]/16
        self.provide_data = data_desc
        self.provide_label = irisIter.provide_label

    def reset(self):
        self.irisIter.reset()
        self.maskIter.reset()

    def iter_next(self):
        if self.irisIter.iter_next() & self.maskIter.iter_next():
            iris = self.irisIter.getdata()
            mask = self.maskIter.getdata()
            iris_label = self.irisIter.getlabel()
            mask_label = self.maskIter.getlabel()
            #assert iris_label.asnumpy() == mask_label.asnumpy()
            tmp_data = mx.nd.concat(iris,mask,dim=1).asnumpy()
            rand_shift = np.random.randint(-self.shift_max,self.shift_max)
            rand_shift = rand_shift if rand_shift>=0 else rand_shift + self.width
            self.cur_data = [mx.nd.array(tmp_data[:,:,:,range(rand_shift,self.width)+range(rand_shift)])]
            self.cur_label = [iris_label]
            return True
        else:
            return False

    def getdata(self):
        return self.cur_data

    def getlabel(self):
        return self.cur_label

    def getpad(self):
        return self.irisIter.getpad()

    def getindex(self):
        return self.irisIter.getindex()
        
        
        

# data
def get_iterator(args, kv):
    kargs = dict(
        data_shape=(1, 64, 512),
    )

    trainIris = mx.io.ImageRecordIter(
        path_imgrec=args.data_dir + 'ice2006iris_train.rec',
        batch_size=args.batch_size,
        rand_crop=False,
        rand_mirror=False,
        num_parts=kv.num_workers,
        part_index=kv.rank,
        shuffle=False,
        **kargs
    )

    trainMask = mx.io.ImageRecordIter(
        path_imgrec=args.data_dir + 'ice2006mask_train.rec',
        batch_size=args.batch_size,
        rand_crop=False,
        rand_mirror=False,
        num_parts=kv.num_workers,
        part_index=kv.rank,
        shuffle=False,
        **kargs
    )

    train = IrisIter(trainIris,trainMask, args.batch_size)

    valIris = mx.io.ImageRecordIter(
        path_imgrec=args.data_dir + 'ice2006iris_test.rec',
        rand_crop=False,
        rand_mirror=False,
        batch_size=args.batch_size,
        num_parts=kv.num_workers,
        part_index=kv.rank,
        **kargs
    )

    valMask = mx.io.ImageRecordIter(
        path_imgrec=args.data_dir + 'ice2006iris_test.rec',
        rand_crop=False,
        rand_mirror=False,
        batch_size=args.batch_size,
        num_parts=kv.num_workers,
        part_index=kv.rank,
        **kargs
    )

    val = IrisIter(valIris,valMask, args.batch_size)
    return (train, val)


class Init(mx.init.Xavier):

    def __call__(self, name, arr):
        """Override () function to do Initialization

        Parameters
        ----------
        name : str
            name of corrosponding ndarray

        arr : NDArray
            ndarray to be Initialized
        """
        if not isinstance(name, mx.base.string_types):
            raise TypeError('name must be string')
        if not isinstance(arr, mx.ndarray.NDArray):
            raise TypeError('arr must be NDArray')
        if 'proj' in name and name.endswith('weight'):
            self._init_proj(name, arr)
        elif name.endswith('upsampling'):
            self._init_bilinear(name, arr)
        elif name.endswith('bias'):
            self._init_bias(name, arr)
        elif name.endswith('gamma'):
            self._init_gamma(name, arr)
        elif name.endswith('beta'):
            self._init_beta(name, arr)
        elif name.endswith('weight'):
            self._init_weight(name, arr)
        elif name.endswith("moving_mean"):
            self._init_zero(name, arr)
        elif name.endswith("moving_var"):
            self._init_zero(name, arr)
        elif name.endswith("moving_inv_var"):
            self._init_zero(name, arr)
        elif name.endswith("moving_avg"):
            self._init_zero(name, arr)
        else:
            self._init_default(name, arr)

    def _init_proj(self, _, arr):
        '''Initialization of shortcut of kenel (2, 2)'''
        w = np.zeros(arr.shape, np.float32)
        for i in range(w.shape[1]):
            w[i, i, ...] = 0.25
        arr[:] = w


class Scheduler(mx.lr_scheduler.MultiFactorScheduler):

    def __init__(self, epoch_step, factor, epoch_size):
        super(Scheduler, self).__init__(
            step=[epoch_size * s for s in epoch_step],
            factor=factor
        )


@mx.optimizer.Optimizer.register
class Nesterov(mx.optimizer.SGD):

    def update(self, index, weight, grad, state):
        """Update the parameters.
        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters
        weight : NDArray
            weight ndarray
        grad : NDArray
            grad ndarray
        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
        """
        assert(isinstance(weight, mx.nd.NDArray))
        assert(isinstance(grad, mx.nd.NDArray))
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = mx.nd.clip(grad, -self.clip_gradient, self.clip_gradient)

        if state[0] is not None:
            mom = state
            mom *= self.momentum
            grad += wd * weight
            mom += grad
            grad += self.momentum * mom
            weight += -lr * grad
        else:
            assert self.momentum == 0.0
            weight += -lr * (grad + self.wd * weight)

    def set_wd_mult(self, args_wd_mult):
        """Set individual weight decay multipler for parameters.
        By default wd multipler is 0 for all params whose name doesn't
        end with _weight, if param_idx2name is provided.

        Parameters
        ----------
        args_wd_mult : dict of string/int to float
            set the wd multipler for name/index to float.
            setting multipler by index is supported for backward compatibility,
            but we recommend using name and symbol.
        """
        self.wd_mult = {}
        for n in self.idx2name.values():
            if not (
                n.endswith('_weight')
                or n.endswith('_bias')
                or n.endswith('_gamma')
                or n.endswith('_beta')
            ) or 'proj' in n or 'zscore' in n:
                self.wd_mult[n] = 0.0
        if self.sym is not None:
            attr = self.sym.attr_dict()
            for k, v in attr.items():
                if k.endswith('_wd_mult'):
                    self.wd_mult[k[:-len('_wd_mult')]] = float(v)
        self.wd_mult.update(args_wd_mult)

    def set_lr_mult(self, args_lr_mult):
        """Set individual learning rate multipler for parameters

        Parameters
        ----------
        args_lr_mult : dict of string/int to float
            set the lr multipler for name/index to float.
            setting multipler by index is supported for backward compatibility,
            but we recommend using name and symbol.
        """
        self.lr_mult = {}
        for n in self.idx2name.values():
            if 'proj' in n or 'zscore' in n:
                self.lr_mult[n] = 0.0
        if self.sym is not None:
            attr = self.sym.attr_dict()
            for k, v in attr.items():
                if k.endswith('_lr_mult'):
                    self.lr_mult[k[:-len('_lr_mult')]] = float(v)
        self.lr_mult.update(args_lr_mult)


def fit(args, network, data_loader, batch_end_callback=None):
    # kvstore
    kv = mx.kvstore.create(args.kv_store)

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    if 'log_file' in args and args.log_file is not None:
        log_file = args.log_file
        log_dir = args.log_dir
        log_file_full_name = os.path.join(log_dir, log_file)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        logger = logging.getLogger()
        handler = logging.FileHandler(log_file_full_name)
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info('start with arguments %s', args)
    else:
        logging.basicConfig(level=logging.DEBUG, format=head)
        logging.info('start with arguments %s', args)

    # load model
    model_prefix = args.model_prefix
    if model_prefix is not None:
        model_prefix += '-%d' % (kv.rank)
    model_args = {}
    if args.load_epoch is not None:
        assert model_prefix is not None
        tmp = mx.model.FeedForward.load(model_prefix, args.load_epoch)
        model_args = {'arg_params': tmp.arg_params,
                      'aux_params': tmp.aux_params,
                      'begin_epoch': args.load_epoch}
    # save model
    save_model_prefix = args.save_model_prefix
    if save_model_prefix is None:
        save_model_prefix = model_prefix
    checkpoint = None if save_model_prefix is None else mx.callback.do_checkpoint(save_model_prefix)

    # data
    (train, val) = data_loader(args, kv)

    # train
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    epoch_size = args.num_examples / args.batch_size

    if args.kv_store == 'dist_sync':
        epoch_size /= kv.num_workers
        model_args['epoch_size'] = epoch_size

    if 'clip_gradient' in args and args.clip_gradient is not None:
        model_args['clip_gradient'] = args.clip_gradient

    # disable kvstore for single device
    if 'local' in kv.type and (
            args.gpus is None or len(args.gpus.split(',')) is 1):
        kv = None

    model = mx.model.FeedForward(
        ctx=devs,
        symbol=network,
        num_epoch=args.num_epochs,
        learning_rate=args.lr,
        momentum=0.9,
        wd=0.0001,
        optimizer='Nesterov',
        initializer=mx.init.Mixed(
            ['.*fc.*', '.*'],
            [mx.init.Xavier(rnd_type='uniform', factor_type='in', magnitude=1),
             Init(rnd_type='gaussian', factor_type='in', magnitude=2)]
        ),
        lr_scheduler=Scheduler(epoch_step=[150, 225], factor=0.1, epoch_size=epoch_size),
        **model_args)

    eval_metrics = ['accuracy']

    if batch_end_callback is not None:
        if not isinstance(batch_end_callback, list):
            batch_end_callback = [batch_end_callback]
    else:
        batch_end_callback = []
    batch_end_callback.append(mx.callback.Speedometer(args.batch_size, 50))
    print train
    model.fit(
        X=train,
        eval_data=val,
        eval_metric=eval_metrics,
        kvstore=kv,
        batch_end_callback=batch_end_callback,
        epoch_end_callback=checkpoint
    )

# train
mx.rnd.seed(2017)

# network
from symbol_densenet import get_symbol
net = get_symbol(712,4,12,12)
fit(args, net, get_iterator)

