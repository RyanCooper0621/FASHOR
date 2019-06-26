%{
Description:
    Fast and scalable higher-order regression model

Reference:


Author:
    Jiaqi Zhang <zjqseu@gmail.com>

History:
    2019/06/26 [Jiaqi Zhang] -- 
%}

function [Err, W] = FASHOR_l1(X, Y, R, lambda, epsilon, iterations, diff)
    %{
    Description:
        Main function for solving FASHOR with L1 norm.
    
    Reference:
        FASHOR
        EE-Ridge

    Inputs:
        X -- observed higher-order samples with shape N * p_1 *...* p_M
        Y -- observed vector responses with shape N
        lambda -- parameter controls the degree of sparsity
        epsilon -- parameter handles with the high-dimensional problem
        iterations -- maximal number of iterations
        diff --  acceptable difference between two generations; 
                 iteration ceases when the difference reaching below this
    
    Outputs:
        Err -- MSE
        W -- estimated higher-order variables
    %}
    % TODO: Variable validation
    %{
    parser=inputParser
    defaultEpsilon = 1.0;
    defaultIter = 100;
    defaultGap = 1e-5;
    addRequired(parser, 'X', @istensor)
    %}
    addpath('tensor_toolbox/');
    Err = 1.0;
    dim = ndims(X);
    XSize = size(X);
    sampleNum = size(Y);
    % initialize weight as a matrix with shape R * (p_1 * p_2 *...* p_M)
    W = zeros(R, prod(XSize(2:end)));
    residual = Y;
    % training
    for r = 1:R
        fprintf('============ The %d-th rank ============\n', r)
        % tensor-form W for calculations
        W_r = vec2Tensor(W(r,:),XSize(2:end));
        % compute the residual
        residual = residual - ttt(X, W_r, 2:dim); 
        % loop until the pre-defined iteration number
        lastW = W(r,:);
        for t = 1:iterations
            fprintf('iteration %d\n', t)
            for m = 2:dim
                % compute Z_{\m}
                % EE selector
            end
            % break when W ceases to improve 
            if norm(W(r,:) - lastW, 'fro') <= diff
                break
            end
        end
    end
    
    
function [res] = vec2Tensor(vec,dimSize)
    startIndex = 1;
    for  m = 1:length(dimSize)
        endIndex = startIndex + dimSize(m);
        tempVec = vec(startIndex:endIndex);
        if m == 1
            res = tempVec;
        else
            res = ttt(res, tempVec);
        end
        startIndex = endIndex + 1;
    end
   
    
    
        
        

