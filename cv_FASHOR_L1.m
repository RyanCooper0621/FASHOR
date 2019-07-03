%{
Description:
    Cross-validation for fast and scalable higher-order regression model.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

History:
    
%}
function [cvLambda, cvEpsilon, cvR] = cv_FASHOR_L1(X, Y, p, lambdaList, epsilonList,...
    RList, fold, iter, diff)
    addpath('tensor_toolbox/')
    if isempty(iter)
        iter = 1000;
    end
    if isempty(diff)
        diff = 1e-4;
    end
    M = length(p);
    %Y = Y.data;
    % initialize the parameter pairs
    parameterPair = {};
    for i = 1:length(lambdaList)
        for j = 1:length(epsilonList)
            for k = 1:length(RList)
                parameterPair{i,j,k} = [lambdaList(i) epsilonList(j) RList(k)];
            end
        end
    end
    % cross-validation initialization  
    N = size(X);
    N = N(1);
    cvp = cvpartition(N, 'Kfold', fold);
    for t = 1:length(lambdaList)*length(epsilonList)*length(RList)
        testErr = 0.0;
        for f = 1:cvp.NumTestSets
            pars = parameterPair{t};
            trains = cvp.training(f);
            tests =  cvp.test(f);
            trainIndex = [];
            testIndex = [];
            for i = 1:N
                if trains(i) == 1
                    trainIndex = [trainIndex i];
                end
                if tests(i) == 1
                    testIndex = [testIndex i];
                end
            end
            % 3-D variates 
            Xtrain = X(trainIndex,:,:,:);
            Ytrain = Y(trainIndex);
            Xtest = X(testIndex,:,:,:);
            Ytest = Y(testIndex);
            [~, estimatedWVec] = FASHOR_L1(Xtrain, Ytrain,...
                        pars(3), pars(1), pars(2), iter, diff);
            % compute estimated coefficients
            for r = 1:pars(3)
                if r == 1
                    estimatedW = vec2Tensor(estimatedWVec(r,:), p);
                else
                    estimatedW = estimatedW + vec2Tensor(estimatedWVec(r,:), p);
                end
            end
            % compute MSE
            predY = ttt(Xtest, estimatedW, 2:M+1, 1:M); 
            testErr = testErr + norm(tensor(predY - Ytest)) / cvp.TestSize(f);
        end
        testErr = testErr / fold;
        % update the best setting of parameters
        if t == 1
            minErr = testErr;
            bestPair = parameterPair{t};
        else
            if testErr < minErr
                minErr = testErr;
                bestPair = parameterPair{t};
            end
        end
    end
    cvLambda = bestPair(1);
    cvEpsilon = bestPair(2);
    cvR = bestPair(3);
    fprintf('cvLambda : %f; cvEpsilon : %f; cvR : %d', cvLambda, cvEpsilon, cvR)
   
        