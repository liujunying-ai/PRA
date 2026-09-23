function [train_outputs, test_outputs] = PL_AGGD(train_data_t,train_data_m,train_p_target,test_data,k,ker,par,Maxiter,lambda,mu,gama)
%PL_AGGD[1] is a partial label learning algorithm 
%    Syntax
%
%       [train_outputs, test_outputs] = PL_AGGD(train_data_t,train_data_m,train_p_target,test_data,k,ker,par,Maxiter,lambda,mu,gama)
%
%    Description
%      
%      parameters,
%           train_data_t   - An m * d array, the ith instance of training instance is stored in train_data_t(i,:), used for predictive model induction
%           train_data_m   - An m * d array, the ith instance of training instance is stored in train_data_m(i,:), used for manifold learning in graph construction
%           train_p_target - An m * q array, if the jth class label is one of the partial labels for the ith training instance, then train_p_target(i,j) equals +1, otherwise train_p_target(i,j) equals 0
%			test_data      - An p * d array, the ith instance of test instance is stored in test_data(i,:) 
%           k              - Number of neighbors,here we set k=10
%           ker            - Type of kernel function,here we use rbf kernel
%           par            - Parameters of kernel function
%      and returns,
%           train_outputs  - An m * q array ,classification results on training data
%           test_outputs   - An p * q array ,classification results on test data
%   [1]D.-B. Wang, L. Li, M.-L. Zhang. Adaptive graph guided disambiguation for partial label learning. In: Proceedings of the 25th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD'19), Anchorage, AK, 2019.
if nargin < 10
	gama = 0.05;
end
if nargin < 9
	mu = 1;
end
if nargin < 8
	lambda = 1;
end
if nargin < 7
	Maxiter = 10;
end
if nargin < 6
	par = 1*mean(pdist(train_data));
end
if nargin < 5
	ker = 'rbf';
end
if nargin < 4
	k = 10;
end
if nargin < 3
	error('Not enough input parameters!');
end
F=build_label_manifold(train_data_m,train_p_target,k);
fprintf('Update parameters...\n')
[train_outputs, test_outputs] = MulRegression(train_data_t, F, test_data, gama, par, ker);
for i = 1:Maxiter
	fprintf('The %d-th iteration\n',i);
	S = obtain_S(train_data_m,F,k,lambda,mu);
	fprintf('Generate the labeling confidence...\n');
	F = UpdateF(S,train_p_target,train_outputs,mu);
	fprintf('Update parameters...\n')
	[train_outputs, test_outputs] = MulRegression(train_data_t, F, test_data, gama, par, ker);
end

end