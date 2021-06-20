function class_gml = classification(ima_summer_rgb, index, labels, polygons)

polygons_drawn = 1;

if polygons_drawn == 0
    %% Read Zurich aerial photos (in Geotiff format)

    % Zurich aerial photos paths
    path(1) = "Data/summer.tif";
    path(2) = "Data/spring_rgb.tif";
    path(3) = "Data/spring_nir.tif";

    display = 1;

    [ima_summer_rgb, ima_summer_nir, ima_spring_rgb, ima_spring_nir] = get_images(path, display);



    %% Trace ROI polygons

    traceROI(ima_summer_rgb);

    return
end


%% supervised classification

image = ima_summer_rgb;

% Reshape the data into a 2d matrix
data = reshape(image,size(image,1)*size(image,2),size(image,3));

% Get pixels from each class mask (polygons)
data_roi = data(cell2mat(index),:);
 
% concatenate the vector of labels
label_roi = [];

for c = 1:length(polygons) % for each polygon
    
    % Create a vector with the label of the polygon class
    % HERE YOUR CODE: label_roi = [label_roi; repmat(labels(c),....
    label_roi = [label_roi; repmat(labels(c),size(index{c},1),1)];
    
end

% Split into training and testing samples to evaluate performances
trainID = 1:10:length(label_roi);
testID = setdiff(1:length(label_roi),trainID);

% Subsample the training and the validation (test) data + labels
data_train = data_roi(trainID,:);
label_train = label_roi(trainID);

data_valid = data_roi(testID,:);
label_valid = label_roi(testID);


%% Rescale the training data using this function classificationScaling

% The following function can rescale the data between [0,1] or that it has
% a unit variance and zero mean
typeNorm = 'minmax'; % use 'std' to rescale to a unit variance and zero mean
[data_train_sc, dataMax, dataMin] = classificationScaling(double(data_train), [], [], typeNorm);

% Rescale accordingly all image pixels
% note: the parameters used on the training pixels are given as arguments in the
% function to rescale accordingly the rest of the pixels
data_sc = classificationScaling(double(data), dataMax, dataMin, typeNorm);

% The same for the validation pixels
data_valid_sc = classificationScaling(double(data_valid), dataMax, dataMin, typeNorm);


%% Training model

% Train a GML model and classifying entire image
class_gml = classify(data_sc,data_train_sc,label_train,'linear');

%% Visualizing classification map

figure

imagesc(label2rgb(reshape(class_gml,size(image,1),size(image,2)),'jet'))
title('GML classification');
axis equal tight
xlabel('x')
ylabel('y')

%% Visualizing classification map
% 
% figure
% subplot(131);
% imagesc(label2rgb(reshape(class_gml,size(image,1),size(image,2)),'jet'))
% title('GML classification');
% axis equal tight
% xlabel('x')
% ylabel('y')

% subplot(132);
% imagesc(label2rgb(reshape(class_knn,size(image,1),size(image,2)),'jet'))
% title(['k-NN classification (k=',int2str(k_knn),')']);
% axis equal tight
% xlabel('x')
% ylabel('y')

% subplot(133);
% imagesc(label2rgb(reshape(class_svm,size(image,1),size(image,2)),'jet'))
% title('SVM classification ');
% axis equal tight
% xlabel('x')
% ylabel('y')





