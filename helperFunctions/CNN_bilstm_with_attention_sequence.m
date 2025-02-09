function lgraph = CNN_bilstm_with_attention_sequence(numFeatures, sequenceLength, filterSize1)
% CNN_CUSTOM_WITH_SELFATTENTION 
%  1D CNN -> SelfAttentionLayer -> BiLSTM -> FC for regression.
%
%  Requires MATLAB R2023b or later (for selfAttentionLayer).
%
%  Example Usage:
%    lgraph = CNN_custom_with_selfAttention(6, 2000, 3, 5);
%    figure; plot(lgraph);
%
%    % Then train:
%    options = trainingOptions("adam", "MaxEpochs", 10, "MiniBatchSize",16);
%    net = trainNetwork(X, Y, lgraph, options);

% -------------------- Create Empty Layer Graph --------------------
lgraph = layerGraph();

%% 1) Input Layer
% Sequence input for a 1D time-series. 
%  - 'MinLength' = sequenceLength ensures the network expects at least that many time steps.
%  - No normalization by default.
inputLayer = sequenceInputLayer(numFeatures, ...
    "Name","input", ...
    "Normalization","none", ...
    "MinLength",sequenceLength);
lgraph = addLayers(lgraph, inputLayer);

%% 2) First Convolutional Block
% 1D convolution along the sequence dimension. 
conv1    = convolution1dLayer(filterSize1, 32, "Padding","same", "Name","conv1");
bn1      = batchNormalizationLayer("Name","bn1");
relu1    = reluLayer("Name","relu1");
maxpool1 = maxPooling1dLayer(2, "Stride",2, "Name","maxpool1");

lgraph = addLayers(lgraph, conv1);
lgraph = addLayers(lgraph, bn1);
lgraph = addLayers(lgraph, relu1);
lgraph = addLayers(lgraph, maxpool1);

% Connect them: input -> conv1 -> bn1 -> relu1 -> maxpool1
lgraph = connectLayers(lgraph, "input",   "conv1");
lgraph = connectLayers(lgraph, "conv1",   "bn1");
lgraph = connectLayers(lgraph, "bn1",     "relu1");
lgraph = connectLayers(lgraph, "relu1",   "maxpool1");

%% 3) Insert Self-Attention Layer
% SelfAttentionLayer (R2023b) automatically expects a sequence format.
% By default, the layer uses 8 heads and a projected dimension of 512 / #heads.
% Adjust 'Heads' or 'KeyDimension', 'ValueDimension' as needed.
%
% For demonstration, let's reduce the heads to 2, 
% so the learnable dimension is smaller (e.g. 64 each).
% You can also set: 'KeyDimension', 'ValueDimension', etc.
selfAttn = selfAttentionLayer(4,16, "Name", "selfAttention");

lgraph = addLayers(lgraph, selfAttn);

% Connect maxpool1 -> selfAttention
lgraph = connectLayers(lgraph, "maxpool1", "selfAttention");

%% 4) BiLSTM Layer
% Takes the sequence output of the selfAttentionLayer and returns the final hidden state.
lstm = bilstmLayer(100, "OutputMode","sequence", "Name","lstm");
lgraph = addLayers(lgraph, lstm);

% Connect selfAttention -> lstm
lgraph = connectLayers(lgraph, "selfAttention", "lstm");


%% 5) Add a global pooling layer
gap = globalAveragePooling1dLayer('Name','gap');

lgraph = addLayers(lgraph,gap);
lgraph = connectLayers(lgraph,"lstm","gap");

%% 6) Output Fully Connected Layers for Regression
fc1 = fullyConnectedLayer(128, "Name","fc1");
fc2 = fullyConnectedLayer(1,   "Name","fc2");  % single scalar output

lgraph = addLayers(lgraph, fc1);
lgraph = addLayers(lgraph, fc2);

% Connect: lstm -> fc1 -> fc2
lgraph = connectLayers(lgraph, "gap","fc1");
lgraph = connectLayers(lgraph, "fc1","fc2");

end
