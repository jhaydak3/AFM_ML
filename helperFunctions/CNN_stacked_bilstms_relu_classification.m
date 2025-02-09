function lgraph = CNN_stacked_bilstms_relu_classification(numFeatures, sequenceLength, filterSize1)
% CNN_custom - Two LSTM layers in "stacked" fashion, fed by a 1D conv block.
%   1) sequenceInputLayer(numFeatures)
%   2) Conv -> BN -> ReLU -> MaxPool
%   3) LSTM (OutputMode="sequence") -- "lstm1"
%   4) LSTM (OutputMode="last")     -- "lstm2"
%   5) FC -> FC (1-dim final)
%
% No regression layer added. Connect 'fc2' externally to a regression or
% other layer as needed.

% ---------------------------
% 1) Initialize layer graph
% ---------------------------
lgraph = layerGraph();

% ---------------------------
% 2) Input Layer
% ---------------------------
inputLayer = sequenceInputLayer(numFeatures, ...
    'Name','input', ...
    'Normalization','none', ...
    'MinLength',sequenceLength);
lgraph = addLayers(lgraph, inputLayer);

% ---------------------------
% 3) First Convolutional Block
% ---------------------------
conv1 = convolution1dLayer(filterSize1, 32, ...
    'Padding','same', ...
    'Name','conv1');
bn1 = batchNormalizationLayer('Name','bn1');
relu1 = reluLayer('Name','relu1');
maxpool1 = maxPooling1dLayer(2, 'Stride', 2, 'Name','maxpool1');

lgraph = addLayers(lgraph, conv1);
lgraph = addLayers(lgraph, bn1);
lgraph = addLayers(lgraph, relu1);
lgraph = addLayers(lgraph, maxpool1);

% Connect the conv block
lgraph = connectLayers(lgraph, 'input','conv1');
lgraph = connectLayers(lgraph, 'conv1','bn1');
lgraph = connectLayers(lgraph, 'bn1','relu1');
lgraph = connectLayers(lgraph, 'relu1','maxpool1');

% ---------------------------
% 4) Stacked LSTMs
% ---------------------------
% LSTM #1: OutputMode = "sequence"
lstm1 = bilstmLayer(100, 'OutputMode','sequence', 'Name','lstm1');
lgraph = addLayers(lgraph, lstm1);

% LSTM #2: OutputMode = "last"
lstm2 = bilstmLayer(100, 'OutputMode','last', 'Name','lstm2');
lgraph = addLayers(lgraph, lstm2);

% Connect stacked LSTMs
%   maxpool1 -> lstm1 -> lstm2
lgraph = connectLayers(lgraph, 'maxpool1','lstm1');
lgraph = connectLayers(lgraph, 'lstm1','lstm2');

% ---------------------------
% 5) Fully Connected Layers
% ---------------------------
fc1 = fullyConnectedLayer(128,'Name','fc1');
lgraph = addLayers(lgraph, fc1);

relu_fc = reluLayer('Name','relu_fc');
lgraph = addLayers(lgraph, relu_fc);

%fc2 = fullyConnectedLayer(1,'Name','fc2');
%lgraph = addLayers(lgraph, fc2);

fc2 = fullyConnectedLayer(2, 'Name', 'fc2');
lgraph = addLayers(lgraph, fc2);

% Add softmax layer
softmax = softmaxLayer('Name', 'softmax');
lgraph = addLayers(lgraph, softmax);



% Connect LSTMs to FC
%   lstm2 -> fc1 -> fc2
lgraph = connectLayers(lgraph, 'lstm2','fc1');
lgraph = connectLayers(lgraph,'fc1','relu_fc');
lgraph = connectLayers(lgraph, 'relu_fc','fc2');
lgraph = connectLayers(lgraph, 'fc2', 'softmax');

end
