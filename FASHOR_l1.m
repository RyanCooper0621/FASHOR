%{
Description:
    Fast and scalable higher-order regression model.

Reference:


Author:
    Jiaqi Zhang <zjqseu@gmail.com>

History:
    2019/06/26 [Jiaqi Zhang] -- Finished coding the functions (FASHOR_l1, 
                                modeProd, softThreshold) and testing them.
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
    dim = ndims(X);
    XSize = size(X);
    % initialize weight as a matrix with shape R * (p_1 * p_2 *...* p_M)
    W = ones(R, sum(XSize(2:end)));
    residual = Y;
    % training
    for r = 1:R
        fprintf('============ The %d-th rank ============\n', r)
        % tensor-form W for calculations
        W_r = vec2Tensor(W(r,:),XSize(2:end));
        % compute the residual
        residual = residual - ttt(X, W_r, 2:dim,1:(dim-1)); 
        clear W_r;
        % loop until the pre-defined iteration number
        lastW = W(r,:);
        last_diff = 100;
        for t = 1:iterations
            fprintf('iteration %d\n', t)
            for m = 2:dim % first dim of X is samples
                % compute Z_{\m}
                Z = modeProd(X, W(r,:), XSize(2:end), m-1);
                %TODO: normalize Z
                Z = Z.data;
                % EE selector
                estimatedW = (Z'*Z+epsilon*eye(XSize(m)))\eye(XSize(m))...
                             * Z' * residual.data;
                estimatedW = softThreshold(estimatedW, lambda);
                startIndex = sum(XSize(2:m-1)) + 1;
                W(r,startIndex:startIndex+XSize(m)-1) = estimatedW;
            end
            % break when W ceases to improve or the improvement is negligible
            cur_diff = norm(W(r,:) - lastW, 'fro')/norm(lastW);
            if cur_diff <= diff || cur_diff > last_diff  
                break
            end
            lastW = W(r,:);
            last_diff = cur_diff;
        end
    end
    Err = norm(residual)/size(Y);
    
    

function [res] = modeProd(X,vecW,dimSize, exclude)
    %{
    Description:
        Cumulative tensor-mode product.

    Inputs:
        X -- tensor observed samples
        vecW -- vectors for each mode
        dimSize -- size of each mode
        exclude -- The index of the mode needs to be excluded in the
                   computation
    
    Outputs:
        res -- product result
    %}
    res = X;
    startIndex = 1;
    countIndex = 1;
    for m = 1:length(dimSize)
        endIndex = startIndex + dimSize(m) - 1;
        if countIndex == exclude
            countIndex = countIndex + 1;
            startIndex = endIndex + 1;
            continue
        end
        tempVec = tensor(vecW(startIndex:endIndex), dimSize(m));
        if countIndex<exclude
            res = ttt(res, tempVec, 2, 1);
        else
            res = ttt(res, tempVec, 3, 1);
        end
        countIndex = countIndex + 1;
        startIndex = endIndex + 1;
    end
    
    
function [res] = softThreshold(vec, threshold)
    %{
    Description:
        Soft-thresholding operator: S_t(x)=sign(x)(|x|-t).

    Inputs:
        vec -- vector needs to be thresholded
        threshold -- the threshold

    Outputs:
        res -- thresholded vector
    %}
    res = vec; 
    for i = 1:length(res)
        if abs(res(i)) <= threshold
            res(i) = 0;
        else
            res(i) = sign(res(i)) * (abs(res(i))-threshold);
        end
    end
            
        
   
    
    
        
        

