% Image processing for Earth Observation
% Project - Schaffhausen Solar Panel
% Version: January 8, 2021
% Author(s): Estelle Droz, Thomas Kimble, Nikitas Papadopoulos

%% Clear everything
clc
clear 
close all

%% Load the image

% Read picture in tif format
file = './Data_zurich/spring_rgb_cut.tif'; 

% Read Geotiff info
info = geotiffinfo(file);

% Create reference matrix
refmat = info.RefMatrix; 

[image_zurich, ~, ~] = geotiffread(file); %All bands


%% Display image

% adjust band intensity values
gamma = 0.5;
% im_adj = imadjust(image_zurich, stretchlim(image_zurich(image_zurich ~= 0), [0.01 0.99]), [0,1], gamma); % adjusted image

for i=1:size(image_zurich,3)
    im_adj(:,:,i) = imadjust(image_zurich(:,:,i), stretchlim(image_zurich(:,:,i), 0.01), [0,1]); % adjusted nir
end

figure
image(im_adj);
axis equal tight
xlabel('col')
ylabel('row')
title('True colour composition')

%% Pre-processing

im_preproc = preprocessing(im_adj, refmat);

%% Trace ROI

traceROI(im_adj);

%% Classification

%load('polygons.mat');
classified_img = classification(im_adj, index, labels, polygons);

%% Post-processing

reshaped_img = reshape(classified_img, size(im_adj, 1:2));

reshaped_img(reshaped_img~=5) = 0;

SE_open = strel('disk',5);
SE_close = strel('disk', 10);
    
% Performing Opening 
im_open = imopen(reshaped_img, SE_open);

% Performing Closing
im_close = imclose(im_open, SE_close);

im_only_roofs = bwareaopen(im_close,2000);

im_only_roofs = imdilate(im_only_roofs, SE_open);

figure
imshow(im_only_roofs);
title('Only roofs');
axis equal tight
xlabel('x')
ylabel('y')

%% Orientation of roofs

stats = regionprops('table', im_only_roofs, 'centroid', 'area', 'orientation')

%% Display only the roofs

im_adj_double= im2double(im_adj);

for i=1:3
    roofs(:,:,i) = im_adj_double(:,:,i).*im_only_roofs;
end

figure
mapshow(roofs, refmat)
axis equal tight
xlabel('col')
ylabel('row')
title('Only roofs')

%% Trying to get every half roof for orientation purposes

smooth_roofs = imguidedfilter(roofs);
im_black = rgb2gray(smooth_roofs);
blackie = watershed(im_black);
imshow(blackie)

%% Confusion matrix

% Reshape the data into a 2d matrix
data = reshape(im_adj,size(im_adj,1)*size(im_adj,2),size(im_adj,3));

% Get pixels from each class mask (polygons) ----- roi indicate masks
% RAPPEL : index?: Nx1 cell array containing the corresponding pixel indices of each ROI polygon
data_roi = data(cell2mat(index),:);
 
% concatenate the vector of labels
label_roi = [];

for c = 1:length(polygons) % for each polygon
    
    % Create a vector with the label of the polygon class
    % HERE YOUR CODE: label_roi = [label_roi; repmat(labels(c),....
    label_roi = [label_roi; repmat(labels(c),size(index{c},1),1)];
    
end

trainID = 1:10:length(label_roi);
testID = setdiff(1:length(label_roi),trainID);

data_train = data_roi(trainID,:);
label_train = label_roi(trainID);

data_valid = data_roi(testID,:);
label_valid = label_roi(testID);

label_valid(label_valid~=5)=1;
label_train(label_train~=5)=1;

typeNorm = 'minmax'; % use 'std' to rescale to a unit variance and zero mean
[data_train_sc, dataMax, dataMin] = classificationScaling(double(data_train), [], [], typeNorm);

% Rescale accordingly all image pixels
% note: the parameters used on the training pixels are given as arguments in the
% function to rescale accordingly the rest of the pixels
data_sc = classificationScaling(double(data), dataMax, dataMin, typeNorm);

% The same for the validation pixels
data_valid_sc = classificationScaling(double(data_valid), dataMax, dataMin, typeNorm);
class_gml = classify(data_sc, data_train_sc, label_train, 'linear');
class_gml_valid = classify(data_valid_sc, data_train_sc, label_train, 'linear');

CT_gml = confusionmat(label_valid, class_gml_valid); % build confusion matrix
OA_gml = trace(CT_gml)/sum(CT_gml(:));
CT_gml_percent=CT_gml./sum(sum(CT_gml));

EA_gml = sum(sum(CT_gml_percent,1)*sum(CT_gml_percent,2));
Ka_gml= (OA_gml - EA_gml)/(1-EA_gml);