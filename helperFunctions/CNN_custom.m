function lgraph = CNN_custom(numFeatures, sequenceLength, filterSize1, filterSize2)

% Initialize layer graph
lgraph = layerGraph();

% Input Layer
inputLayer = sequenceInputLayer(numFeatures, 'Name', 'input', 'Normalization', 'none', 'MinLength', sequenceLength);
lgraph = addLayers(lgraph, inputLayer);

% First Convolutional Block
conv1 = convolution1dLayer(filterSize1, 32, 'Padding', 'same', 'Name', 'conv1');
bn1 = batchNormalizationLayer('Name', 'bn1');
relu1 = reluLayer('Name', 'relu1');
maxpool1 = maxPooling1dLayer(2, 'Stride', 2, 'Name', 'maxpool1');

lgraph = addLayers(lgraph, conv1);
lgraph = addLayers(lgraph, bn1);
lgraph = addLayers(lgraph, relu1);
lgraph = addLayers(lgraph, maxpool1);

lgraph = connectLayers(lgraph, 'input', 'conv1');
lgraph = connectLayers(lgraph, 'conv1', 'bn1');
lgraph = connectLayers(lgraph, 'bn1', 'relu1');
lgraph = connectLayers(lgraph, 'relu1', 'maxpool1');


% LSTM Layer
lstm = lstmLayer(100, 'OutputMode', 'last', 'Name', 'lstm');
lgraph = addLayers(lgraph, lstm);
% lgraph = connectLayers(lgraph, 'self_attention', 'lstm');
lgraph = connectLayers(lgraph, 'maxpool1', 'lstm');


% Output Fully Connected Layer
fc1 = fullyConnectedLayer(128, 'Name', 'fc1');
lgraph = addLayers(lgraph, fc1);
lgraph = connectLayers(lgraph, 'lstm', 'fc1');


fc2 = fullyConnectedLayer(1, 'Name', 'fc2');
lgraph = addLayers(lgraph, fc2);
%lgraph = connectLayers(lgraph, 'lstm', 'fc');
lgraph = connectLayers(lgraph, 'fc1', 'fc2');
end
