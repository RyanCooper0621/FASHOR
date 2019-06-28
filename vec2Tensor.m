%{
Description:
    Revert the tensor through its CP decomposed tensor factors. 

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

History:
    2019/06/27 [Jiaqi Zhang]-- Finished coding the function (vec2Tensor)
                               and testing it. 
%}

function [res] = vec2Tensor(vec,dimSize)
    %{
    Description:
        Revert the tensor through its CP decomposed tensor factors. 
        The size of vector is p_1*p_2*...*p_M. 

    Inputs:
        vec -- A vector of tensor factors with shape p_1*p_2*...*p_M..
        dimSize -- A vector of integers denotes the size of each
                   dimension respectively.

    Outputs:
        res -- Reverted tensor.
    %}
    addpath('tensor_toolbox/');
    startIndex = 1;
    for  m = 1:length(dimSize)
        endIndex = startIndex + dimSize(m) - 1;
        tempVec = vec(startIndex:endIndex);
        tempVec=tensor(tempVec,dimSize(m));
        if m == 1
            res = tempVec;
        else
            % vector outer product
            res = ttt(res, tempVec);
        end
        startIndex = endIndex + 1;
    end

    
