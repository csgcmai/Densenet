clear; close all;

db_name = 'frgc';

im_path = sprintf('/dev/shm/csgcmai/img_%s_pca_rec_rbf.mat', db_name);
of_path = sprintf('/dev/shm/csgcmai/rec_pca_rec_rbf_openface_%s.mat', db_name);
ori_of_path = sprintf('~/project/database/lfwfrgc/ori_openface_%s.mat',db_name);
configFile = sprintf('../config/%s/blufr_%s_config.mat', db_name, db_name);

rof = h5read(of_path,'/openface');
of = h5read(ori_of_path,'/openface');
load(configFile);
n_img = size(of,2);

score_10 = [];
for i_test = 1:numel(trainIndex)
    s = diag(of'*squeeze(rof(:,:,i_test)));
    s(trainIndex{i_test}) = -1;
    score_10 = [score_10, s(:)];
end

[max_score_10, max_score_i] = max(score_10');
score = max_score_10(:);

openface = zeros(128,n_img);
data = zeros(96,96,3,n_img,'uint8');
load(im_path)

for i_img = 1:n_img
    openface(:,i_img) = rof(:,i_img,max_score_i(i_img));
    data(:,:,:,i_img) = rec_datas(:,:,:,i_img,max_score_i(i_img));
    imwrite(permute(squeeze(data(:,:,:,i_img)),[2 1 3]), sprintf('/dev/shm/csgcmai/rec_img_%s/%d.png',db_name,i_img));
end

save(sprintf('../data/img_%s_pca_rec_rbf_best.mat',db_name), 'data','openface','score','-v7.3');    



