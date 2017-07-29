%% This demo shows how to apply PCA dimension reduction along with 
%the cosine similarity measure for the BLUFR benchmark and report results.
close all; clear; clc;

dbname = 'lfw';

% Mat file storing raw images
imgFile = sprintf('/home/comp/csgcmai/project/database/lfwfrgc/img_%s.mat',dbname); 
openfaceFile = sprintf('/home/comp/csgcmai/project/database/lfwfrgc/ori_openface_%s.mat',dbname); 
% configuration file for this evaluation
configFile = sprintf('../config/%s/blufr_%s_config.mat', dbname, dbname); 
outDir = '../result/'; % output directory

outMatFile = [outDir, 'img_', dbname, '_pca_rec_rbf.mat']; % output mat file
outLogFile = [outDir, 'img_', dbname, '_pca_rec_rbf.txt']; % output txt file

tic;
fprintf('Load data...\n\n');
load(configFile);

%% Load your own features here. The features should be extracted according
% to the order of the imageList in the configFile. It is 13233xd for the 
% LFW database where d is the feature dimensionality.
load(imgFile, 'data');
openface = h5read(openfaceFile,'/openface');

data1 = data;
data = double(permute(data,[4 1 2 3]));
data = reshape(data,size(data,1), 96*96*3);

openface = permute(openface,[2 1]);

numTrials = length(testIndex);

rec_datas = zeros(96,96,3,size(data,1),10,'uint8');

%% Calculate w*pinv(w') with 10 trials.
for t = 1 : numTrials
    fprintf('Process the %dth trial...\n\n', t);
    
    pcaDims = numel(trainIndex{t})-1; % PCA dimensions.
    % Get the training data of the t'th trial.
    of_tr = openface(trainIndex{t},:);
    X_tr = data(trainIndex{t}, :);

    of_tr_mean = mean(of_tr);
    of_tr = bsxfun(@minus, of_tr, of_tr_mean);

    % for the computation of phi, it is noted that: sqrt(s^2+(a-b)^2) 
    % = sqrt(1+a^2+b^2-2ab)=sqrt(3-2ab), where openface feature are l2 normalized. 
    phi = sqrt(3-2*of_tr*of_tr');
    
    % Learn a PCA subspace. Note that if you apply a learning based dimension 
    % reduction, it must be performed with the training data of each trial. 
    % It is not allowed to learn and reduce the dimensionality of features
    % with the whole data beforehand and then do the 10-trial evaluation.
    [W,pca_tr] = PCA(X_tr,pcaDims);
    w_rbf = pinv(phi)*pca_tr;
    
    of_te = openface; %obtain the input tempalte
    of_te = bsxfun(@minus,openface, of_tr_mean); %process as the way of training data
    of_te(trainIndex{t},:) = 0; %set the training data as zero

    % *** marked!!! we are not sure that whether the centering is applied on the testing data here!!! ***
    phi_te = sqrt(3-2*of_te*of_tr');
    pca_te_rbf = phi_te*w_rbf;

    pinvW = pinv(W);
    X_rec = pca_te_rbf*pinvW;
    
    X_rec(trainIndex{t},:) = 0;
    % add back the mean data
    X_rec = bsxfun(@plus, X_rec,mean(data));
    X_rec(X_rec>255) = 255;
    X_rec(X_rec<0) = 0;
    X_rec = uint8(X_rec);
    X_rec = reshape(X_rec,size(X_rec,1),96,96,3);
    X_rec = permute(X_rec, [2 3 4 1]);
    
    rec_datas(:,:,:,:,t) = X_rec;
end

%save(outMatFile,'rec_datas', '-v7.3');

