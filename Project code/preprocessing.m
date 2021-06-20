function im_preproc = preprocessing(im_adj, refmat)
    
    %% Convert outputs to double 
    im_adj= im2double(im_adj);
    
    %% Use of edge preserving blurring filter
    im_aniso = imguidedfilter(im_adj);
    
    %% Normalization function to use within mapshow
    imnorm = @(x) (x - min(x(:))) ./ (max(x(:)) - min(x(:)));
  
    %% Contour reinforcement with high-pass filter
    cont_filt = [-0.5 -0.5 -0.5 -0.5 -0.5; -0.5 -0.5 -0.5 -0.5 -0.5; ...
             -0.5 -0.5 13 -0.5 -0.5; -0.5 -0.5 -0.5 -0.5 -0.5; -0.5 -0.5 -0.5 -0.5 -0.5];
    im_contour = imfilter(im_aniso, cont_filt);

    %% Dilation and erosion
    
    SE_1 = strel('disk',1);
    
    im_contour = im2uint8(im_contour);
    
    % Performing dilation
    im_dilate = imdilate(im_contour, SE_1);
    
    % Performing erosion
    im_ero = imerode(im_dilate, SE_1);

    %% Opening and closing
    
    SE_2 = strel('disk',4);
    
    % Performing Opening 
    im_open = imopen(im_ero, SE_2);
    
    % Performing Closing
    im_close = imclose(im_open, SE_2);
    
    %% Second anisotropic filter and contour reinforcement
    
    im_close= im2double(im_close);
    
    im_aniso2 = imguidedfilter(im_close);
    
    im_cont2 = imsharpen(im_aniso2);
    
    figure
    mapshow(im_cont2(:,:,:), refmat)
    axis equal tight
    xlabel('easting [m]')
    ylabel('northing [m]')
    title('Final preprocessed image')   
    
    %% Sobel filtering
    hsobel = fspecial('sobel');
    im_sobel_h = imfilter(im_close, hsobel);

    % To detect in the other direction: transpose the filter
    im_sobel_v = imfilter(im_close, hsobel');

    % To combine directions: L2-norm of the two directions (use sqrt() and
    % power())
    im_sobel_iso = sqrt(power(im_sobel_h,2)+power(im_sobel_v,2));
    
    figure
    % plot isotropic Sobel
    mapshow(imnorm(im_sobel_iso(:,:,:)), refmat)
    axis equal tight
    xlabel('easting [m]')
    ylabel('northing [m]')
    title('Sobel isotropic')
 
    %% Output of preprocessed image
    
    im_preproc = im_cont2;
    
end