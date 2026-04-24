%% RHFNet Image Classification Script
% Former Author: JoeyBG.
% Improved By: JoeyBG.
% Date: 2026-04-07.
% Affiliate: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Introduction:
%   This script implements the RHFNet based on FasterNet-T0 architecture
%       for radar human activity recognition image classification.
%   It utilizes a custom Deep Learning Layer to execute the Partial Convolution (PConv)
%       to reduce redundant computations and memory access.
%   The dataset is loaded and split randomly into an 8:2 ratio for training and validation.
%   No additional data augmentation is applied to the image data.
%   The script trains the model using the AdamW/Adam optimizer and tracks the process.
%   Finally, it exports customized visualizations for accuracy, loss curves, and 
%       the validation confusion matrix using the predefined visual parameter style.
%
% References:
%   [1] J. Chen et al., "Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks," 
%       in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023.

%% Preparation for Matlab Script
close all;
clear all;
clc;
disp("---------- © Author: JoeyBG © ----------");

%% Definition
% Define dataset path and basic training parameters
Dataset_Path = 'SimH_RTM_Dataset';
Num_Classes = 12;                                                           % Number of activity labels in total
Batch_Size = 32;                                                            % Training batch size of the network
Max_Epochs = 20;                                                            % Maximum training epochs
Initial_LR = 0.00147;                                                       % Initial learning rate                            

% Define the visualization style parameters
Font_Name = 'Palatino Linotype';                                            
Font_Size_Basis = 17;                                                       
Font_Size_Axis = 20;                                                        
Font_Size_Title = 22;                                                       
Font_Weight_Basis = 'normal';                                               
Font_Weight_Axis = 'normal';                                                
Font_Weight_Title = 'bold';                                                 

% Define the custom colormap colors for the visualization plots
JoeyBG_Colormap =[0.6196 0.0039 0.2588;                                     
                   0.8353 0.2431 0.3098;
                   0.9569 0.4275 0.2627;
                   0.9922 0.6824 0.3804;
                   0.9961 0.8784 0.5451;
                   1.0000 1.0000 0.7490;
                   0.9020 0.9608 0.5961;
                   0.6706 0.8667 0.6431;
                   0.4000 0.7608 0.6471;
                   0.1961 0.5333 0.7412;
                   0.3686 0.3098 0.6353];                                   % My favorite colormap for visualization                                   

%% Dataset Loading & Splitting
disp("Start loading dataset and spliting...");

% Load images from the specified folder
imds = imageDatastore(Dataset_Path, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% 8:2 split for Train and Validation sets
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');

% Resize images to fit the network input requirement [224, 224, 3] without aug
augimdsTrain = augmentedImageDatastore([224 224 3], imdsTrain);
augimdsVal = augmentedImageDatastore([224 224 3], imdsVal);

disp(['Training images: ', num2str(numel(imdsTrain.Files))]);
disp(['Validation images: ', num2str(numel(imdsVal.Files))]);

%% Build RHFNet LayerGraph
disp("Constructing RHFNet architecture...");

lgraph = layerGraph();

% 1. Stem
stem_layers = [
    imageInputLayer([224 224 3], 'Normalization', 'none', 'Name', 'input')
    convolution2dLayer(4, 40, 'Stride', 4, 'Padding', 0, 'Name', 'stem_conv')
    batchNormalizationLayer('Name', 'stem_bn')
    geluLayer('Name', 'stem_gelu')
];
lgraph = addLayers(lgraph, stem_layers);
lastLayer = 'stem_gelu';

% Specific embedding params
embed_dims = [40, 80, 160, 320];
depths = [1, 2, 8, 2];

% 2. Stages
for i = 1:4
    % Downsampling for stages 2, 3, 4
    if i > 1
        ds_name = sprintf('ds_%d', i);
        ds_layers = [
            convolution2dLayer(2, embed_dims(i), 'Stride', 2, 'Padding', 0, 'Name', [ds_name '_conv'])
            batchNormalizationLayer('Name', [ds_name '_bn'])
            geluLayer('Name', [ds_name '_gelu'])
        ];
        lgraph = addLayers(lgraph, ds_layers);
        lgraph = connectLayers(lgraph, lastLayer, [ds_name '_conv']);
        lastLayer = [ds_name '_gelu'];
    end
    
    % Residual Blocks
    for j = 1:depths(i)
        bname = sprintf('stage%d_blk%d', i, j);
        
        blk_layers = [
            PConvLayer(embed_dims(i), 4, 'Name', [bname '_pconv'])
            convolution2dLayer(1, embed_dims(i)*2, 'Name', [bname '_conv1'])
            batchNormalizationLayer('Name', [bname '_bn'])
            geluLayer('Name', [bname '_gelu'])
            convolution2dLayer(1, embed_dims(i), 'Name', [bname '_conv2'])
        ];
        add_layer = additionLayer(2, 'Name', [bname '_add']);
        
        lgraph = addLayers(lgraph, blk_layers);
        lgraph = addLayers(lgraph, add_layer);
        
        lgraph = connectLayers(lgraph, lastLayer, [bname '_pconv']);
        lgraph = connectLayers(lgraph, [bname '_conv2'], [bname '_add/in1']);
        lgraph = connectLayers(lgraph, lastLayer, [bname '_add/in2']);
        
        lastLayer = [bname '_add'];
    end
end

% 3. Head
head_layers = [
    globalAveragePooling2dLayer('Name', 'gap')
    convolution2dLayer(1, 1280, 'Name', 'head_conv')
    geluLayer('Name', 'head_gelu')
    fullyConnectedLayer(Num_Classes, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')
];
lgraph = addLayers(lgraph, head_layers);
lgraph = connectLayers(lgraph, lastLayer, 'gap');

% Analyze the constructed network model
analyzeNetwork(lgraph);

%% Train Network
disp("Start Training Process...");

options = trainingOptions('adam', ...
    'InitialLearnRate', Initial_LR, ...
    'MaxEpochs', Max_Epochs, ...
    'MiniBatchSize', Batch_Size, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsVal, ...
    'ValidationFrequency', floor(numel(imdsTrain.Files)/Batch_Size), ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'ExecutionEnvironment', 'auto', ...
    'OutputNetwork', 'best-validation');

[net, info] = trainNetwork(augimdsTrain, lgraph, options);

disp("Training Completed.");

%% Plot Results Analysis
disp("Plotting Accuracy, Loss Curves and Confusion Matrix...");

iterations = 1:length(info.TrainingLoss);
val_idx = ~isnan(info.ValidationLoss); % Find iterations where validation occurred

% Loss Curve Visualization
fig_loss = figure('Name', 'Training & Validation Loss', 'Position', [100, 100, 700, 500]);
ax_loss = axes('Parent', fig_loss);
hold(ax_loss, 'on'); grid(ax_loss, 'on');
plot(ax_loss, iterations, info.TrainingLoss, 'Color', JoeyBG_Colormap(end-1,:), 'LineWidth', 2, 'DisplayName', 'Training Loss');
plot(ax_loss, iterations(val_idx), info.ValidationLoss(val_idx), 'Color', JoeyBG_Colormap(2,:), 'LineWidth', 2, 'DisplayName', 'Validation Loss');
set(ax_loss, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'FontWeight', Font_Weight_Basis);
% title('Loss Curve', 'FontName', Font_Name, 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
xlabel('Training steps', 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis);
ylabel('Cross Entropy Loss', 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis);
legend(ax_loss, 'Location', 'northeast');

% Accuracy Curve Visualization
fig_acc = figure('Name', 'Training & Validation Accuracy', 'Position', [150, 150, 700, 500]);
ax_acc = axes('Parent', fig_acc);
hold(ax_acc, 'on'); grid(ax_acc, 'on');
plot(ax_acc, iterations, info.TrainingAccuracy, 'Color', JoeyBG_Colormap(end-1,:), 'LineWidth', 2, 'DisplayName', 'Training Accuracy');
plot(ax_acc, iterations(val_idx), info.ValidationAccuracy(val_idx), 'Color', JoeyBG_Colormap(2,:), 'LineWidth', 2, 'DisplayName', 'Validation Accuracy');
set(ax_acc, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'FontWeight', Font_Weight_Basis);
% title('Accuracy Curve', 'FontName', Font_Name, 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
xlabel('Training steps', 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis);
ylabel('Accuracy (%)', 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis);
ylim(ax_acc, [0, 100]);
legend(ax_acc, 'Location', 'southeast');

% Confusion Matrix with plotconfusion
disp("Evaluating Validation Set...");
YPred = classify(net, augimdsVal);
YVal = imdsVal.Labels;

% Transform categorical labels to dummy variable matrix required by plotconfusion
T_Val_Mat = dummyvar(double(YVal))';
Y_Pred_Mat = dummyvar(double(YPred))';

fig_conf = figure('Name', 'Confusion Matrix Validation', 'Position', [200, 200, 800, 800]);
plotconfusion(T_Val_Mat, Y_Pred_Mat);
ax_conf = gca;
class_labels = arrayfun(@(x) sprintf('S%d', x), 1:Num_Classes, 'UniformOutput', false);
current_xticks = get(ax_conf, 'XTickLabel');
current_yticks = get(ax_conf, 'YTickLabel');
current_xticks(1:Num_Classes) = class_labels;
current_yticks(1:Num_Classes) = class_labels;
set(ax_conf, 'XTickLabel', current_xticks, 'YTickLabel', current_yticks);
set(ax_conf, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'FontWeight', Font_Weight_Basis);
title('', 'FontName', Font_Name, 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
xlabel('Target Class', 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis);
ylabel('Output Class', 'FontName', Font_Name, 'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis);
text_handles = findall(fig_conf, 'Type', 'Text');
set(text_handles, 'FontName', Font_Name, 'FontSize', Font_Size_Basis);