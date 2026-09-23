function [ACC_PRA, ACC_Base, train_outputs, test_outputs] = PRA_PL_AGGD(X_train,y_train,X_test,y_test, k,ker,par,Maxiter,lambda,mu,gamma)
%PRA_PL_AGGD implements the PRA framework presented in [1] with PL-AGGD as the base learner
%
%    Syntax
%
%       [ACC_PRA, ACC_Base, train_outputs, test_outputs] = PRA_PL_AGGD(X_train,y_train,X_test,y_test, k,ker,par,Maxiter,lambda,mu,gamma)
%
%    Description
%
%       PRA_PL_AGGD takes,
%           X_train	- A MxD array, the ith instance of training instance is stored in X_train(i,:)
%           y_train - A MxQ array, if the jth class label is one of the partial labels for the ith training instance, then y_train(i,j)=+1, otherwise y_train(i,j)=0
%           X_test  - A PxD array, the ith instance of testing instance is stored in X_test(i,:)
%           y_test  - A PxQ array, if the ith test instance belongs to the jth class, then y_test(i,j)=+1, otherwise y_test(i,j)=0
%      and returns,
%           ACC_PRA  - A scalar, the accuracy of PRA
%           ACC_Base - A scalar, the accuracy of base learner
%           train_outputs - A MxQ array ,classification results on training data
%           test_outputs  - A PxQ array ,classification results on test data
%
%  [1] J.-Y. Liu, J.-P. Sun, Y.-H. Zhao, B.-B. Jia. Augmenting Graph-Based Partial Label Learning with Predictive Representations. Electronics, 2026, 15(18), Article 4210. https://doi.org/10.3390/electronics15184210 

    %include the PL-AGGD package, avaiable at: https://palm.seu.edu.cn/zhangml/files/PL-AGGD.rar
    addpath('PL_AGGD');%Note that we have made some necessary modifications to the software package.
    
    %default parameters setting, the same as the default parameters of the base learner
    if nargin < 4
        error('At least four parameters need to be provided, namely X_train, y_train, X_test and y_test.');
    end
    if nargin < 11 || isempty(gamma)
        gamma = 0.05;%In fact, it is the lambda in PL-AGGD
    end
    if nargin < 10 || isempty(mu)
        mu = 1;
    end
    if nargin < 9  || isempty(lambda)
        lambda = 1;%In fact, it is the gamma in PL-AGGD
    end
    if nargin < 8  || isempty(Maxiter)
        Maxiter = 10;
    end
    if nargin < 7  || isempty(par)
        par = 1*mean(pdist(X_train));%Parameters of kernel function
    end
    if nargin < 6  || isempty(ker)
        ker = 'rbf';%Type of kernel function
    end
    if nargin < 5  || isempty(k)
        k = 10;%Number of neighbors
    end
    
    %Initial training to generate predictive representations (train_outputs1 & test_outputs1)
    [train_outputs1, test_outputs1] = PL_AGGD(X_train,X_train,y_train,X_test,k,ker,par,Maxiter,lambda,mu,gamma);
    ACC_Base = CalAccuracy(test_outputs1, y_test);
    
    %Standardize predictive representations of training set (train_outputs1)
    [train_outputs1_zscore,mu_zscore,sigma_zscore] = zscore(train_outputs1);
    
    %Standardize predictive representations of test set (test_outputs1) using training set statistics
    tmp_minus = bsxfun(@minus, test_outputs1, mu_zscore);
    test_outputs1_zscore = bsxfun(@rdivide, tmp_minus, sigma_zscore);
    
    %The concatenation of the original raw features and the standardized predictive representation is used for prediction model induction
    X_train_t = [X_train,train_outputs1_zscore];
    X_test_t = [X_test,test_outputs1_zscore];
    
    %The standardized predictive representation is used for the manifold learning in graph construction 
    X_train_m = train_outputs1_zscore;
    
    %Update the parameters of kernel function
    par = 1*mean(pdist(X_train_t));
    
    %Augmented retraining
    [train_outputs, test_outputs] = PL_AGGD(X_train_t,X_train_m,y_train,X_test_t,k,ker,par,Maxiter,lambda,mu,gamma);
    ACC_PRA = CalAccuracy(test_outputs, y_test);
    rmpath('PL_AGGD');
end

