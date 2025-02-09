function lgraph = CNN_Sotres_classification(numFeatures,sequenceLength)
% CNN_Sotres Create a simple 1D CNN for time-series data (one feature, length=sequenceLength).
% 
% This network is very similar to the original Python code with 3 Conv1D + MaxPool blocks,
% but includes a globalAveragePooling1dLayer in MATLAB to collapse the leftover time dimension.
% 
% Why this layer?
% In TensorFlow/Keras, a Flatten layer automatically merges all spatial/time dimensions
% into a single feature vector before the Dense layer, so there's no leftover "time" dimension.
% In MATLAB's sequence mode, however, we must explicitly remove the extra time dimension
% (e.g., by global average pooling). Otherwise, MATLAB might keep a [C x B x T] shape
% through the final layer, leading to dimension mismatches with the scalar targets.

kernelSize = 9;
%==================== Initialize layer graph ====================
lgraph = layerGraph();

%==================== Input Layer ====================
% Single-feature input, up to 'sequenceLength' time steps
inputLayer = sequenceInputLayer(numFeatures, ...
    'Name','input', ...
    'Normalization','none', ...
    'MinLength', sequenceLength);
lgraph = addLayers(lgraph, inputLayer);

%==================== First Convolutional Block ====================
conv1    = convolution1dLayer(kernelSize, 32, ...
    'Padding','same', ...
    'Name','conv1d_1');
relu1    = reluLayer('Name','relu1');
maxpool1 = maxPooling1dLayer(2, 'Stride',2, 'Name','maxPool_1');

lgraph = addLayers(lgraph, conv1);
lgraph = addLayers(lgraph, relu1);
lgraph = addLayers(lgraph, maxpool1);

lgraph = connectLayers(lgraph,'input','conv1d_1');
lgraph = connectLayers(lgraph,'conv1d_1','relu1');
lgraph = connectLayers(lgraph,'relu1','maxPool_1');

%==================== Second Convolutional Block ====================
conv2    = convolution1dLayer(kernelSize, 64, ...
    'Padding','same', ...
    'Name','conv1d_2');
relu2    = reluLayer('Name','relu2');
maxpool2 = maxPooling1dLayer(2, 'Stride',2, 'Name','maxPool_2');

lgraph = addLayers(lgraph, conv2);
lgraph = addLayers(lgraph, relu2);
lgraph = addLayers(lgraph, maxpool2);

lgraph = connectLayers(lgraph,'maxPool_1','conv1d_2');
lgraph = connectLayers(lgraph,'conv1d_2','relu2');
lgraph = connectLayers(lgraph,'relu2','maxPool_2');

%==================== Third Convolutional Block ====================
conv3    = convolution1dLayer(kernelSize, 128, ...
    'Padding','same', ...
    'Name','conv1d_3');
relu3    = reluLayer('Name','relu3');
maxpool3 = maxPooling1dLayer(2, 'Stride',2, 'Name','maxPool_3');

lgraph = addLayers(lgraph, conv3);
lgraph = addLayers(lgraph, relu3);
lgraph = addLayers(lgraph, maxpool3);

lgraph = connectLayers(lgraph,'maxPool_2','conv1d_3');
lgraph = connectLayers(lgraph,'conv1d_3','relu3');
lgraph = connectLayers(lgraph,'relu3','maxPool_3');

%==================== Global Average Pool + Fully Connected Layers ====================
% We replace the Flatten layer with globalAveragePooling1dLayer to mimic how Python's Flatten
% merges all spatial/time dims. This ensures that the final FC gets a [Channels x Batch]
% shape instead of [Channels x Batch x leftoverTime].
gap  = globalAveragePooling1dLayer('Name','gap');
fc1  = fullyConnectedLayer(128,'Name','Dense');
fc2  = fullyConnectedLayer(2,  'Name','fc2');  % single scalar for regression
softmax = softmaxLayer('Name', 'softmax');

lgraph = addLayers(lgraph, gap);
lgraph = addLayers(lgraph, fc1);
lgraph = addLayers(lgraph, fc2);
lgraph = addLayers(lgraph,softmax);

lgraph = connectLayers(lgraph, 'maxPool_3','gap');
lgraph = connectLayers(lgraph, 'gap','Dense');
lgraph = connectLayers(lgraph, 'Dense','fc2');
lgraph = connectLayers(lgraph,'fc2','softmax');

end
