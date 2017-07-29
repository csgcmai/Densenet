%% This demo shows how to apply a supervised learning method for 
% the BLUFR benchmark and report results. Also, it shows how to learn
% optimal parameters using the development set.
close all; clear; 

dbname = 'casiav4thousand';
% dbname = 'iitd';
epoch = 10;
feaFile = ['/home/comp/csgcmai/project/database/iris/',dbname,'_ice06-e',num2str(epoch,'%03d'),'.mat']; % Mat file storing extracted features. Replace this with your own feature.
outDir = './result/'; % output directory
outMatFile = [outDir, 'result_', dbname,'_ice06-e',num2str(epoch,'%03d'),'.mat']; % output mat file
outLogFile = [outDir, 'result_', dbname,'_ice06-e',num2str(epoch,'%03d'),'.txt']; % output text file

veriFarPoints = [0, kron(10.^(-8:-1), 1:9), 1]; % FAR points for face verification ROC plot
reportVeriFar = 0.001; % the FAR point for verification performance reporting

[~, veriFarIndex] = ismember(reportVeriFar, veriFarPoints);
%% Load your own features here. The features should be extracted according
% to the order of the imageList in the configFile. It is 13233xd for the 
% LFW database where d is the feature dimensionality.

% You may apply the sqrt transform if the feature is histogram based.
Descriptors = h5read(feaFile,'/feature');
labels = h5read(feaFile,'/label');

testX = normr(Descriptors');
score = testX * testX'; 
testLabels = labels;
[VR, veriFAR] = EvalROC(score, testLabels, [], veriFarPoints);

fprintf('Verification:\n');
fprintf('\t@ FAR = %g%%: VR = %g%%.\n', reportVeriFar*100, VR(veriFarIndex)*100);

%% Plot the face verification ROC curve.
figure; semilogx(veriFAR * 100, VR*100, 'LineWidth', 2);
xlim([0,100]); ylim([0,100]); grid on;
xlabel('False Accept Rate (%)');
ylabel('Verification Rate (%)');
title('Face Verification ROC Curve');

%save(outMatFile, 'reportVeriFar', 'reportOsiFar', 'reportRank', 'reportVR', 'reportDIR', ...
%    'meanVeriFAR', 'fusedVR', 'meanOsiFAR', 'fusedDIR', 'rankPoints', 'rankIndex', 'osiFarIndex');
