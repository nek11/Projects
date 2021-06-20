function [d] =  compute_distance(x_1, x_2, type)
%COMPUTE_DISTANCE Computes the distance between two datapoints (as column vectors)
%   depending on the choosen distance type={'L1','L2','LInf'}
%
%   input -----------------------------------------------------------------
%   
%       o x_1   : (N x 1),  N-dimensional datapoint
%       o x_2   : (N x 1),  N-dimensional datapoint
%       o type  : (string), type of distance {'L1','L2','LInf'}
%
%   output ----------------------------------------------------------------
%
%       o d      : distance between x_1 and x_2 depending on distance
%                  type {'L1','L2','LInf'}
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch type
    case "L1"
        d = sum(abs(x_2-x_1));
    case "L2"
        d = sqrt(sum((x_2-x_1).^2));
    case "LInf"
        d = max(abs(x_2-x_1));
end

end