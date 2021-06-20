function [ima_summer_rgb, ima_summer_nir, ima_spring_rgb, ima_spring_nir] = get_images(path, display)
% VISUALISE_IMAGES This function returns reads paths and returns image
% matrices. This function also displays images if wanted.
%
% input
% - path: file paths in the following order: summer, spring rgb, spring nir
% - display: boolean that displays images if true
%
% output:
% - ima_summer_rgb
% - ima_summer_nir
% - ima_spring_rgb
% - ima_spring_nir


%% Read images

% Summer: 4 bands (R,G,B,NIR)
[ima_summer_raw, ~, refmat_summer, ~] = geotiffread(path(1));
info_summer = geotiffinfo(path(1));

% Spring RGB: 3 bands (R,G,B)
[ima_spring_rgb_raw, ~, refmat_spring_rgb, ~] = geotiffread(path(2));

% Spring NIR: 3 bands (NIR,R,G)
[ima_spring_nir_raw, ~, refmat_spring_nir, ~] = geotiffread(path(3));


%% Get location

% Get latitude and longitude
[x,y] = pix2map(info_summer.RefMatrix, 1, 1);
[lat,lon] = projinv(info_summer, x,y);

% Print latitude and longitude
fprintf('Location: %f N, %f E.\n', lat, lon)


%% Create different color composites

% Summer: True colors adjusted 3 band intensity values (contrast stretch 
% to bottom and top 1% of pixel values)
ima_summer_rgb(:,:,1) = imadjust(ima_summer_raw(:,:,1));
ima_summer_rgb(:,:,2) = imadjust(ima_summer_raw(:,:,2));
ima_summer_rgb(:,:,3) = imadjust(ima_summer_raw(:,:,3));

% Summer: NIR, R and G adjusted 3 band intensity values (contrast stretch 
% to bottom and top 1% of pixel values)
ima_summer_nir(:,:,1) = imadjust(ima_summer_raw(:,:,4));
ima_summer_nir(:,:,2) = ima_summer_rgb(:,:,1);
ima_summer_nir(:,:,3) = ima_summer_rgb(:,:,2);

% Spring: True colors adjusted 3 band intensity values (contrast stretch 
% to bottom and top 1% of pixel values)
ima_spring_rgb(:,:,1) = imadjust(ima_spring_rgb_raw(:,:,1));
ima_spring_rgb(:,:,2) = imadjust(ima_spring_rgb_raw(:,:,2));
ima_spring_rgb(:,:,3) = imadjust(ima_spring_rgb_raw(:,:,3));

% Spring: NIR, R and G adjusted 3 band intensity values (contrast stretch 
% to bottom and top 1% of pixel values)
ima_spring_nir(:,:,1) = imadjust(ima_spring_nir_raw(:,:,1));
ima_spring_nir(:,:,2) = imadjust(ima_spring_nir_raw(:,:,2));
ima_spring_nir(:,:,3) = imadjust(ima_spring_nir_raw(:,:,3));


%% Plot images

if display == 1
    figure

    subplot(2,2,1)
    mapshow(ima_summer_rgb, refmat_summer, 'DisplayType', 'image')
    axis equal tight
    xlabel('x')
    ylabel('y')
    title('Summer: Adjusted true colors (R,G,B)')

    subplot(2,2,3)
    mapshow(ima_summer_nir, refmat_summer, 'DisplayType', 'image')
    axis equal tight
    xlabel('x')
    ylabel('y')
    title('Summer: Adjusted color composite (NIR,R,G)')

    subplot(2,2,2)
    mapshow(ima_spring_rgb, refmat_spring_rgb, 'DisplayType', 'image')
    axis equal tight
    xlabel('x')
    ylabel('y')
    title('Spring: Adjusted true colors (R,G,B)')

    subplot(2,2,4)
    mapshow(ima_spring_nir, refmat_spring_nir, 'DisplayType', 'image')
    axis equal tight
    xlabel('x')
    ylabel('y')
    title('Spring: Adjusted color composite (NIR,R,G)')
end


end

