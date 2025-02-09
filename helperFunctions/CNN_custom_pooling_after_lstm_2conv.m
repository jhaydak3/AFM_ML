function lgraph = CNN_custom_pooling_after_lstm_2conv(numFeatures, sequenceLength, filterSize)
% CNN_custom_pooling_after_lstm:
%   1) One Convolution block
%   2) LSTM with 'OutputMode="sequence"'
%   3) Global average pooling
%   4) FullyConnected -> ReLU -> FullyConnected
% No second conv block. No regression layer included.

    % 1) Initialize layer graph
    lgraph = layerGraph();

    % 2) Input layer
    inputLayer = sequenceInputLayer(numFeatures, ...
        'Name','input', ...
        'Normalization','none', ...
        'MinLength',sequenceLength);
    lgraph = addLayers(lgraph, inputLayer);

    % 3) Convolution + BN + ReLU + MaxPool
    conv1 = convolution1dLayer(filterSize, 32, 'Padding','same', 'Name','conv1');
    lgraph = addLayers(lgraph, conv1);

    bn1 = batchNormalizationLayer('Name','bn1');
    lgraph = addLayers(lgraph, bn1);

    relu1 = reluLayer('Name','relu1');
    lgraph = addLayers(lgraph, relu1);

    mp1 = maxPooling1dLayer(2,'Stride',2,'Name','maxpool1');
    lgraph = addLayers(lgraph, mp1);

    % 3b) Second conv block

    conv2 = convolution1dLayer(filterSize, 32, 'Padding','same', 'Name','conv2');
    lgraph = addLayers(lgraph, conv2);

    bn2 = batchNormalizationLayer('Name','bn2');
    lgraph = addLayers(lgraph, bn2);

    relu2 = reluLayer('Name','relu2');
    lgraph = addLayers(lgraph, relu2);

    mp2 = maxPooling1dLayer(2,'Stride',2,'Name','maxpool2');
    lgraph = addLayers(lgraph, mp2);

    % 4) LSTM (OutputMode="sequence")
    lstmSeq = lstmLayer(100,'OutputMode','sequence','Name','lstmSeq');
    lgraph = addLayers(lgraph, lstmSeq);

    % 5) Global average pooling
    gap = globalAveragePooling1dLayer('Name','globalPool');
    lgraph = addLayers(lgraph, gap);

    % 6) Fully connected + ReLU + final FC (scalar output)
    fc1 = fullyConnectedLayer(128,'Name','fc1');
    lgraph = addLayers(lgraph, fc1);

    % reluFC1 = reluLayer('Name','relu_fc1');
    % lgraph = addLayers(lgraph, reluFC1);

    fcOut = fullyConnectedLayer(1,'Name','fc_out');
    lgraph = addLayers(lgraph, fcOut);

    % 7) Connect the layers in a chain
    lgraph = connectLayers(lgraph, 'input','conv1');
    lgraph = connectLayers(lgraph, 'conv1','bn1');
    lgraph = connectLayers(lgraph, 'bn1','relu1');
    lgraph = connectLayers(lgraph, 'relu1','maxpool1');

    lgraph = connectLayers(lgraph, 'maxpool1','conv2');
    lgraph = connectLayers(lgraph, 'conv2','bn2');
    lgraph = connectLayers(lgraph, 'bn2','relu2');
    lgraph = connectLayers(lgraph, 'relu2','maxpool2');
    
    lgraph = connectLayers(lgraph, 'maxpool2','lstmSeq');
    lgraph = connectLayers(lgraph, 'lstmSeq','globalPool');
    lgraph = connectLayers(lgraph, 'globalPool','fc1');
    %lgraph = connectLayers(lgraph, 'fc1','relu_fc1');
    lgraph = connectLayers(lgraph, 'fc1','fc_out');

end
