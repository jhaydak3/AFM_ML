function lgraph = CNN_Sotres_ResNet50_classification(numFeatures,inputLength, kernelSizeStage1, kernelSizeBlocks, nOut)
% CNN_Sotres_ResNet50 - A fully manual 1D ResNet-50 style network with no auto-chaining.
%
% Stages:
%   (1) Input -> Conv(stride=2) -> BN -> ReLU -> MaxPool(3,stride=2)
%   (2) convBlock(stride=2) -> 2x identityBlock
%   (3) convBlock(stride=2) -> 3x identityBlock
%   (4) convBlock(stride=2) -> 5x identityBlock
%   (5) convBlock(stride=2) -> 2x identityBlock
% Final:
%   globalAveragePooling1dLayer -> fullyConnectedLayer(nOut)
%
% inputLength       e.g. 2000
% kernelSizeStage1  e.g. 3
% kernelSizeBlocks  e.g. 3
% nOut              e.g. 1 => scalar regression

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 0) Create empty layerGraph
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lgraph = layerGraph();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1) Stage 1: Input -> Conv(stride=2) -> BN -> ReLU -> MaxPool(3,stride=2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Add each layer individually + connect once to avoid duplicates

% A) sequenceInput
inp = sequenceInputLayer(numFeatures, ...
    'Name','input',...
    'Normalization','none',...
    'MinLength',inputLength);
lgraph = addLayers(lgraph, inp);

% B) conv1
conv1 = convolution1dLayer(kernelSizeStage1, 64, ...
    'Name','conv1', ...
    'Padding',3, ...
    'Stride',2);
bn1   = batchNormalizationLayer('Name','bn_conv1');
relu1 = reluLayer('Name','relu_conv1');
pool1 = maxPooling1dLayer(3, 'Stride',2, 'Name','pool1');

lgraph = addLayers(lgraph, conv1);
lgraph = addLayers(lgraph, bn1);
lgraph = addLayers(lgraph, relu1);
lgraph = addLayers(lgraph, pool1);

% C) Connect Stage 1
lgraph = connectLayers(lgraph, 'input','conv1');
lgraph = connectLayers(lgraph,'conv1','bn_conv1');
lgraph = connectLayers(lgraph,'bn_conv1','relu_conv1');
lgraph = connectLayers(lgraph,'relu_conv1','pool1');

lastStage = 'pool1';  % We'll feed Stage 2 from here

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2) Stage 2: convBlock(stride=2) + 2 identityBlocks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[lgraph,lastStage] = addConvBlock1D(lgraph, lastStage, ...
    'convBlock2a', [16,16,64], kernelSizeBlocks, 'Stride',2);

[lgraph,lastStage] = addIdentityBlock1D(lgraph, lastStage, ...
    'idBlock2b', [16,16,64], kernelSizeBlocks);

[lgraph,lastStage] = addIdentityBlock1D(lgraph, lastStage, ...
    'idBlock2c', [16,16,64], kernelSizeBlocks);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3) Stage 3: convBlock(stride=2) + 3 identityBlocks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[lgraph,lastStage] = addConvBlock1D(lgraph, lastStage, ...
    'convBlock3a', [32,32,128], kernelSizeBlocks, 'Stride',2);

[lgraph,lastStage] = addIdentityBlock1D(lgraph, lastStage, ...
    'idBlock3b', [32,32,128], kernelSizeBlocks);
[lgraph,lastStage] = addIdentityBlock1D(lgraph, lastStage, ...
    'idBlock3c', [32,32,128], kernelSizeBlocks);
[lgraph,lastStage] = addIdentityBlock1D(lgraph, lastStage, ...
    'idBlock3d', [32,32,128], kernelSizeBlocks);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4) Stage 4: convBlock(stride=2) + 5 identityBlocks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[lgraph,lastStage] = addConvBlock1D(lgraph,lastStage, ...
    'convBlock4a',[64,64,256], kernelSizeBlocks,'Stride',2);

[lgraph,lastStage] = addIdentityBlock1D(lgraph,lastStage, ...
    'idBlock4b',[64,64,256], kernelSizeBlocks);
[lgraph,lastStage] = addIdentityBlock1D(lgraph,lastStage, ...
    'idBlock4c',[64,64,256], kernelSizeBlocks);
[lgraph,lastStage] = addIdentityBlock1D(lgraph,lastStage, ...
    'idBlock4d',[64,64,256], kernelSizeBlocks);
[lgraph,lastStage] = addIdentityBlock1D(lgraph,lastStage, ...
    'idBlock4e',[64,64,256], kernelSizeBlocks);
[lgraph,lastStage] = addIdentityBlock1D(lgraph,lastStage, ...
    'idBlock4f',[64,64,256], kernelSizeBlocks);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5) Stage 5: convBlock(stride=2) + 2 identityBlocks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[lgraph,lastStage] = addConvBlock1D(lgraph,lastStage, ...
    'convBlock5a',[64,64,256], kernelSizeBlocks,'Stride',2);

[lgraph,lastStage] = addIdentityBlock1D(lgraph,lastStage, ...
    'idBlock5b',[64,64,256], kernelSizeBlocks);
[lgraph,lastStage] = addIdentityBlock1D(lgraph,lastStage, ...
    'idBlock5c',[64,64,256], kernelSizeBlocks);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 6) Final: globalAveragePooling1dLayer -> fullyConnectedLayer(nOut)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gap   = globalAveragePooling1dLayer('Name','gap');
fcOut = fullyConnectedLayer(nOut,'Name','fc-out');
softmax = softmaxLayer('Name','softmax');
lgraph = addLayers(lgraph, gap);
lgraph = addLayers(lgraph, fcOut);
lgraph = addLayers(lgraph, softmax);

% Connect final
lgraph = connectLayers(lgraph, lastStage, 'gap');
lgraph = connectLayers(lgraph, 'gap','fc-out');
lgraph = connectLayers(lgraph,'fc-out','softmax');

end % main CNN_Sotres_ResNet50
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                 SUBFUNCTION: addConvBlock1D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [lgraph, outName] = addConvBlock1D(lgraph, inName, prefix, ...
    filters, kernelSize, varargin)
% addConvBlock1D - A "conv_block" for 1D ResNet: 3-layer main path + 1-layer shortcut => add => ReLU
%
% Usage:
%   [lgraph,outName] = addConvBlock1D(lgraph, 'pool1', 'convBlock2a', ...
%       [16,16,64], 3, 'Stride',2);

p = inputParser();
addParameter(p,'Stride',2,@isscalar);
parse(p,varargin{:});
strideVal = p.Results.Stride;

[f1,f2,f3] = deal(filters(1), filters(2), filters(3));

% MAIN path
convA = convolution1dLayer(1,f1,'Name',[prefix '_2a'],'Stride',strideVal);
bnA   = batchNormalizationLayer('Name',[prefix '_bn2a']);
reluA = reluLayer('Name',[prefix '_relu2a']);

convB = convolution1dLayer(kernelSize,f2,'Padding','same','Name',[prefix '_2b']);
bnB   = batchNormalizationLayer('Name',[prefix '_bn2b']);
reluB = reluLayer('Name',[prefix '_relu2b']);

convC = convolution1dLayer(1,f3,'Name',[prefix '_2c']);
bnC   = batchNormalizationLayer('Name',[prefix '_bn2c']);

% SHORTCUT path
convS = convolution1dLayer(1,f3,'Stride',strideVal,'Name',[prefix '_1']);
bnS   = batchNormalizationLayer('Name',[prefix '_bn1']);

% ADD + final ReLU
addL = additionLayer(2,'Name',[prefix '_add']);
reluF= reluLayer('Name',[prefix '_out']);

% Add each layer individually
lgraph = addLayers(lgraph, convA);
lgraph = addLayers(lgraph, bnA);
lgraph = addLayers(lgraph, reluA);

lgraph = addLayers(lgraph, convB);
lgraph = addLayers(lgraph, bnB);
lgraph = addLayers(lgraph, reluB);

lgraph = addLayers(lgraph, convC);
lgraph = addLayers(lgraph, bnC);

lgraph = addLayers(lgraph, convS);
lgraph = addLayers(lgraph, bnS);

lgraph = addLayers(lgraph, addL);
lgraph = addLayers(lgraph, reluF);

% Connect main path
lgraph = connectLayers(lgraph, inName,       [prefix '_2a']);
lgraph = connectLayers(lgraph,[prefix '_2a'],[prefix '_bn2a']);
lgraph = connectLayers(lgraph,[prefix '_bn2a'],[prefix '_relu2a']);
lgraph = connectLayers(lgraph,[prefix '_relu2a'],[prefix '_2b']);
lgraph = connectLayers(lgraph,[prefix '_2b'],[prefix '_bn2b']);
lgraph = connectLayers(lgraph,[prefix '_bn2b'],[prefix '_relu2b']);
lgraph = connectLayers(lgraph,[prefix '_relu2b'],[prefix '_2c']);
lgraph = connectLayers(lgraph,[prefix '_2c'],[prefix '_bn2c']);

% Shortcut path
lgraph = connectLayers(lgraph, inName,[prefix '_1']);
lgraph = connectLayers(lgraph,[prefix '_1'],[prefix '_bn1']);

% add => final ReLU
lgraph = connectLayers(lgraph,[prefix '_bn2c'], [prefix '_add' '/in1']);
lgraph = connectLayers(lgraph,[prefix '_bn1'],  [prefix '_add' '/in2']);
lgraph = connectLayers(lgraph,[prefix '_add'],  [prefix '_out']);

outName = [prefix '_out'];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%              SUBFUNCTION: addIdentityBlock1D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [lgraph, outName] = addIdentityBlock1D(lgraph, inName, ...
    prefix, filters, kernelSize, varargin)
% addIdentityBlock1D - A "no stride" block: 3-layer main path + direct skip => add => ReLU
%
% Usage:
%   [lgraph,outName] = addIdentityBlock1D(lgraph, 'pool1', 'idBlock2b', ...
%       [16,16,64], 3);

p = inputParser();
addParameter(p,'Stride',1,@isscalar);
parse(p,varargin{:});
strideVal = p.Results.Stride;  %#ok<NASGU> (If you want to apply stride in identity blocks)

[f1,f2,f3] = deal(filters(1), filters(2), filters(3));

convA = convolution1dLayer(1,f1,'Name',[prefix '_2a']);
bnA   = batchNormalizationLayer('Name',[prefix '_bn2a']);
reluA = reluLayer('Name',[prefix '_relu2a']);

convB = convolution1dLayer(kernelSize,f2,...
    'Padding','same',...
    'Name',[prefix '_2b']);
bnB   = batchNormalizationLayer('Name',[prefix '_bn2b']);
reluB = reluLayer('Name',[prefix '_relu2b']);

convC = convolution1dLayer(1,f3,'Name',[prefix '_2c']);
bnC   = batchNormalizationLayer('Name',[prefix '_bn2c']);

addL = additionLayer(2,'Name',[prefix '_add']);
reluF= reluLayer('Name',[prefix '_out']);

% Add them
lgraph = addLayers(lgraph, convA);
lgraph = addLayers(lgraph, bnA);
lgraph = addLayers(lgraph, reluA);

lgraph = addLayers(lgraph, convB);
lgraph = addLayers(lgraph, bnB);
lgraph = addLayers(lgraph, reluB);

lgraph = addLayers(lgraph, convC);
lgraph = addLayers(lgraph, bnC);

lgraph = addLayers(lgraph, addL);
lgraph = addLayers(lgraph, reluF);

% main path
lgraph = connectLayers(lgraph, inName,       [prefix '_2a']);
lgraph = connectLayers(lgraph,[prefix '_2a'],[prefix '_bn2a']);
lgraph = connectLayers(lgraph,[prefix '_bn2a'],[prefix '_relu2a']);
lgraph = connectLayers(lgraph,[prefix '_relu2a'],[prefix '_2b']);
lgraph = connectLayers(lgraph,[prefix '_2b'],[prefix '_bn2b']);
lgraph = connectLayers(lgraph,[prefix '_bn2b'],[prefix '_relu2b']);
lgraph = connectLayers(lgraph,[prefix '_relu2b'],[prefix '_2c']);
lgraph = connectLayers(lgraph,[prefix '_2c'],[prefix '_bn2c']);

% skip path = direct
lgraph = connectLayers(lgraph, inName, [prefix '_add' '/in2']);
lgraph = connectLayers(lgraph,[prefix '_bn2c'], [prefix '_add' '/in1']);

lgraph = connectLayers(lgraph,[prefix '_add'], [prefix '_out']);

outName = [prefix '_out'];
end
