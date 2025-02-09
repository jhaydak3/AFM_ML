function lgraph = CNN_Sotres(numFeatures, sequenceLength)
% CNN_Sotres2  Create a "1D CNN" by treating the input as a [Height x Width x Channels] image.
%   - imageInputLayer([sequenceLength 1 numFeatures])  => "height = sequenceLength"
%   - We do 2D convolutions with filterSize=[kernelSize 1]
%   - Use flattenLayer before fullyConnectedLayer
%   - The final layer is a regressionLayer for scalar output

kernelSize = 7;

% Build as an array of layers for simplicity
layers = [
    imageInputLayer([sequenceLength 1 numFeatures], ...
        'Name','input',...
        'Normalization','none')  % No 'DataFormat' needed here
    
    % ----- First Convolutional Block -----
    convolution2dLayer([kernelSize 1], 32, 'Padding','same','Name','conv1')
    reluLayer('Name','relu1')
    maxPooling2dLayer([2 1], 'Stride',[2 1], 'Name','maxPool_1')
    
    % ----- Second Convolutional Block -----
    convolution2dLayer([kernelSize 1], 64, 'Padding','same','Name','conv2')
    reluLayer('Name','relu2')
    maxPooling2dLayer([2 1], 'Stride',[2 1], 'Name','maxPool_2')
    
    % ----- Third Convolutional Block -----
    convolution2dLayer([kernelSize 1], 128, 'Padding','same','Name','conv3')
    reluLayer('Name','relu3')
    maxPooling2dLayer([2 1], 'Stride',[2 1], 'Name','maxPool_3')
    
    % ----- Flatten + Fully Connected -----
    flattenLayer('Name','flatten')
    fullyConnectedLayer(128, 'Name','fc1')
    %reluLayer('Name','relu4')  % an extra ReLU if you want
    fullyConnectedLayer(1, 'Name','fc2')  % single scalar
];

% Convert the array into a layer graph
lgraph = layerGraph(layers);

% If you want to see the connectivity visually:
% analyzeNetwork(lgraph)
end
