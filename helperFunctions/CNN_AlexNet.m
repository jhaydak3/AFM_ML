function lgraph = CNN_AlexNet(numFeatures, sequenceLength)
% CNN_AlexNet_1DHeightOnly
%
% Treat your 1D data of size [sequenceLength x numFeatures]
% as a 2D image of [sequenceLength, numFeatures, 1],
% but only convolve/pool over the sequence dimension (height).
%
% KERNELS (conv/pool) are [k x 1].
%  - conv1: [11 x 1], stride [4 x 1], pad [5 0 5 0]
%  - conv2: [5  x 1], grouped, pad [2 0 2 0]
%  - conv3: [3  x 1], pad [1 0 1 0]
%  - ...
%  - pool layers: [3 x 1], stride [2 x 1], *NO padding* => [0 0 0 0]
%
% This avoids "pooling region dimensions must be greater than the padding
% dimensions" by removing pooling padding.
%
% Example:
%   lgraph = CNN_AlexNet_1DHeightOnly(6,2000);
%   figure; plot(lgraph);
%   dlnet = dlnetwork(lgraph);
%   dlnet = initialize(dlnet); 
%   % Then train with trainNetwork(X, Y, lgraph, options).

%% 1) Input layer
inputLayer = imageInputLayer([sequenceLength, numFeatures, 1], ...
    'Name','data');

%% 2) Define conv/pool layers (height-only) + AlexNet heads

% --- conv1: [11 x 1], stride [4 x 1], pad top/bottom=5, left/right=0
conv1 = convolution2dLayer([11 1],96, ...
    'Name','conv1', ...
    'Stride',[4 1], ...
    'Padding',[5 0 5 0], ...
    'BiasLearnRateFactor',2);
relu1 = reluLayer('Name','relu1');
norm1 = crossChannelNormalizationLayer(5,'Name','norm1','K',1);

% --- pool1: [3 x 1], stride [2 x 1], NO padding
pool1 = maxPooling2dLayer([3 1], ...
    'Name','pool1',...
    'Stride',[2 1],...
    'Padding',[0 0 0 0]);

% --- conv2 (grouped): [5 x 1], pad top/bottom=2
conv2 = groupedConvolution2dLayer([5 1],128,2, ...
    'Name','conv2',...
    'Padding',[2 0 2 0],...
    'BiasLearnRateFactor',2);
relu2 = reluLayer('Name','relu2');
norm2 = crossChannelNormalizationLayer(5,'Name','norm2','K',1);

% --- pool2: [3 x 1], stride [2 x 1], NO padding
pool2 = maxPooling2dLayer([3 1], ...
    'Name','pool2',...
    'Stride',[2 1],...
    'Padding',[0 0 0 0]);

% --- conv3: [3 x 1], pad top/bottom=1
conv3 = convolution2dLayer([3 1],384,...
    'Name','conv3',...
    'Padding',[1 0 1 0],...
    'BiasLearnRateFactor',2);
relu3 = reluLayer('Name','relu3');

% --- conv4 (grouped): [3 x 1], pad top/bottom=1
conv4 = groupedConvolution2dLayer([3 1],192,2, ...
    'Name','conv4',...
    'Padding',[1 0 1 0],...
    'BiasLearnRateFactor',2);
relu4 = reluLayer('Name','relu4');

% --- conv5 (grouped): [3 x 1], pad top/bottom=1
conv5 = groupedConvolution2dLayer([3 1],128,2, ...
    'Name','conv5',...
    'Padding',[1 0 1 0],...
    'BiasLearnRateFactor',2);
relu5 = reluLayer('Name','relu5');

% --- pool5: [3 x 1], stride [2 x 1], NO padding
pool5 = maxPooling2dLayer([3 1], ...
    'Name','pool5',...
    'Stride',[2 1],...
    'Padding',[0 0 0 0]);

% --- fc6 -> relu6 -> drop6
fc6   = fullyConnectedLayer(4096,'Name','fc6','BiasLearnRateFactor',2);
relu6 = reluLayer('Name','relu6');
drop6 = dropoutLayer(0.5,'Name','drop6');

% --- fc7 -> relu7 -> drop7
fc7   = fullyConnectedLayer(4096,'Name','fc7','BiasLearnRateFactor',2);
relu7 = reluLayer('Name','relu7');
drop7 = dropoutLayer(0.5,'Name','drop7');

% --- fc8 -> fc9
fc8   = fullyConnectedLayer(1000,'Name','fc8','BiasLearnRateFactor',2);
fc9 = fullyConnectedLayer(1,'Name','fc9','BiasLearnRateFactor',2);



%% 3) Combine in a single chain
layers = [
    inputLayer
    conv1
    relu1
    norm1
    pool1

    conv2
    relu2
    norm2
    pool2

    conv3
    relu3
    conv4
    relu4
    conv5
    relu5
    pool5

    fc6
    relu6
    drop6
    fc7
    relu7
    drop7

    fc8
    fc9
    ];

%% 4) Convert to layer graph (sequential; no extra connections)
lgraph = layerGraph(layers);

end
