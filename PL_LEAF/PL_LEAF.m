function [ train_outputs,test_outputs,accuracy ] = PL_LEAF( X_train_t,X_train_m,y_train,X_test,y_test,k,ker,C1,C2,epsi,par,tol )
    [Beta,b,train_outputs] =PL_LEAF_train(X_train_t,X_train_m,y_train,k,ker,C1,C2,epsi,par,tol);
    [accuracy,test_outputs]= PL_LEAF_predict(X_train_t,X_test,y_test,ker,Beta,b,par);
end

