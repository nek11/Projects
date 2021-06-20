function [cimg, ApList, muList] = compress_image(img, p)
%COMPRESS_IMAGE Compress the image by applying the PCA over each channels 
% independently
%
%   input -----------------------------------------------------------------
%   
%       o img : (width x height x 3), an image of size width x height over RGB channels
%       o p : The number of components to keep during projection 
%
%   output ----------------------------------------------------------------
%
%       o cimg : (3 x p x height) The projection of the image on the eigenvectors
%       o ApList : (3 x p x width) The projection matrices for each channel
%       o muList : (3 x width x 1) The mean vector for each channel


X1 = img(:,:,1);
[mu1,~,EigenVectors1,~] = compute_pca(X1);
[cimg(1,:,:), ApList(1,:,:)] = project_pca(X1, mu1, EigenVectors1, p);

X2 = img(:,:,2);
[mu2,~,EigenVectors2,~] = compute_pca(X2);
[cimg(2,:,:), ApList(2,:,:)] = project_pca(X2, mu2, EigenVectors2, p);

X3 = img(:,:,3);
[mu3,~,EigenVectors3,~] = compute_pca(X3);
[cimg(3,:,:), ApList(3,:,:)] = project_pca(X3, mu3, EigenVectors3, p);

muList(1,:,:) = mu1;
muList(2,:,:) = mu2;
muList(3,:,:) = mu3;


end

