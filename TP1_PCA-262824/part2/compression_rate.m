function [cr, compressedSize] = compression_rate(img,cimg,ApList,muList)
%COMPRESSION_RATE Calculate the compression rate based on the original
%image and all the necessary components to reconstruct the compressed image
%
%   input -----------------------------------------------------------------
%       o img : The original image   
%       o cimg : The compressed image
%       o ApList : List of projection matrices for each independent
%       channels
%       o muList : List of mean vectors for each independent channels
%
%   output ----------------------------------------------------------------
%
%       o cr : The compression rate
%       o compressedSize : The size of the compressed elements

s_img = 64*numel(img);
s_ap = 64*numel(ApList);
s_y = 64*numel(cimg);
s_mu = 64*numel(muList);

cr = 1 - (s_y+s_ap+s_mu)/s_img;

compressedSize = s_ap+s_mu+s_y;


% convert the size to megabits
compressedSize = compressedSize / 1048576; 
end

