%This is an exemplar file on how the PRA framework with PL-CL as the base learner could be used
%Type 'help PRA_PL_CL' under Matlab prompt for more detailed information
clear;clc;close all;fclose('all');
%More data sets are publicly available at: 
%https://palm.seu.edu.cn/zhangml/Resources.htm#partial_data
load('MSRCv2.mat');%load the partial label data set
X_load = zscore(data);%help zscore
y_load_p = transpose(full(partial_target));%partial target
y_load_r = transpose(full(target));%real target

numFolds = 10;%ten-fold cross validation
numInstances = size(X_load,1);
rng(1,'v5uniform');%rand('state', 1);
idx_rand = randperm(numInstances);

ACC_PRA = zeros(numFolds,1);
ACC_Base = zeros(numFolds,1);
for numFold=1:numFolds
    temp_str = ['Fold-', num2str(numFold), ' begins...'];
    disp(temp_str);
    %split dataset into training set and testing set
    [idx_train,idx_test] = CV_data_partition(numInstances,numFolds,numFold);
    X_train = X_load(idx_rand(idx_train),:);
    y_train = y_load_p(idx_rand(idx_train),:);
    X_test = X_load(idx_rand(idx_test),:);
    y_test = y_load_r(idx_rand(idx_test),:);
    %train & test
    [ACC_PRA(numFold), ACC_Base(numFold)] = PRA_PL_CL(X_train,y_train,X_test,y_test);
end
temp_str = [ ' PRA: ACC = ', num2str(mean(ACC_PRA),'%4.3f'),'¡À', num2str(std(ACC_PRA),'%4.3f')];disp(temp_str);
temp_str = [ 'Base: ACC = ', num2str(mean(ACC_Base),'%4.3f'),'¡À', num2str(std(ACC_Base),'%4.3f')];disp(temp_str);
