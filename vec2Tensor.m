%{
Description:
    Convert a vector into a tensor, given the size of each mode 
    respectively.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

History:
    2019/06/27 [Jiaqi Zhang]-- Finished coding the function (vec2Tensor)
                               and testing it. 
%}

function [res] = vec2Tensor(vec,dimSize)
    %{
    Description:
        Convert a vector into a tensor, given the size of each dimension.

    Inputs:
        vec -- The vector needs to converted.
        dimSize -- A sequence of integers denotes the size of each
                   dimension respectively.

    Outputs:
        res -- Converted tensor.
    %}
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

    
