function lgraph = CNN_Sotres_ResNet50(numFeatures, inputLength, ...
    kernelSizeStage1, kernelSizeBlocks, nOut)
% CNN_Sotres_ResNet50 - ResNet-50 style network but using 2D layers
% to handle a 1D time dimension as an "image" of size [inputLength x 1].
%
% Stages (similar to your original):
%   (1) Input -> Conv(stride=2) -> BN -> ReLU -> MaxPool(3,stride=2)
%   (2) convBlock(stride=2) -> 2x identityBlock
%   (3) convBlock(stride=2) -> 3x identityBlock
%   (4) convBlock(stride=2) -> 5x identityBlock
%   (5) convBlock(stride=2) -> 2x identityBlock
% Final:
%   averagePooling2dLayer -> flattenLayer -> fullyConnectedLayer(nOut)
%
% Inputs:
%   numFeatures    e.g. 1 (channels)
%   inputLength    e.g. 2000 (signal length)
%   kernelSizeStage1 e.g. 3
%   kernelSizeBlocks e.g. 3
%   nOut           e.g. 1 => scalar output (add a regressionLayer if desired)

lgraph = layerGraph();

%% 1) Input layer as 2D image: [Height x Width x Channels]
%    "Height"     = inputLength
%    "Width"      = 1
%    "Channels"   = numFeatures
inputLayer = imageInputLayer([inputLength 1 numFeatures], ...
    'Name','input',...
    'Normalization','none');  % No 'DataFormat' needed

lgraph = addLayers(lgraph,inputLayer);

%% 2) Stage 1: Conv(stride=2) -> BN -> ReLU -> MaxPool(3,stride=2)
% We'll emulate "Padding=3" by `[3 0]` in 2D, and "Stride=2" by `[2 1]`
conv1 = convolution2dLayer([kernelSizeStage1 1], 64, ...
    'Name','conv1',...
    'Padding',[3 0], ...
    'Stride',[2 1]);  % stride 2 in time dimension only
bn1   = batchNormalizationLayer('Name','bn_conv1');
relu1 = reluLayer('Name','relu_conv1');
pool1 = maxPooling2dLayer([3 1],'Name','pool1',...
    'Padding',[1 0], ...  % optional to preserve dimension or mimic original
    'Stride',[2 1]);      % stride 2 in time dimension

lgraph = addLayers(lgraph,conv1);
lgraph = addLayers(lgraph,bn1);
lgraph = addLayers(lgraph,relu1);
lgraph = addLayers(lgraph,pool1);

% Connect them in sequence
lgraph = connectLayers(lgraph,'input','conv1');
lgraph = connectLayers(lgraph,'conv1','bn_conv1');
lgraph = connectLayers(lgraph,'bn_conv1','relu_conv1');
lgraph = connectLayers(lgraph,'relu_conv1','pool1');

lastStage = 'pool1';  % We'll feed Stage 2 from here

%% 3) Stage 2: convBlock(stride=2) + 2 identityBlocks
[lgraph,lastStage] = addConvBlock2D(lgraph,lastStage,...
    'convBlock2a',[16,16,64], [kernelSizeBlocks 1], 'Stride',[2 1]);

[lgraph,lastStage] = addIdentityBlock2D(lgraph,lastStage,...
    'idBlock2b',[16,16,64],[kernelSizeBlocks 1]);
[lgraph,lastStage] = addIdentityBlock2D(lgraph,lastStage,...
    'idBlock2c',[16,16,64],[kernelSizeBlocks 1]);

%% 4) Stage 3: convBlock(stride=2) + 3 identityBlocks
[lgraph,lastStage] = addConvBlock2D(lgraph,lastStage,...
    'convBlock3a',[32,32,128],[kernelSizeBlocks 1],'Stride',[2 1]);

[lgraph,lastStage] = addIdentityBlock2D(lgraph,lastStage,...
    'idBlock3b',[32,32,128],[kernelSizeBlocks 1]);
[lgraph,lastStage] = addIdentityBlock2D(lgraph,lastStage,...
    'idBlock3c',[32,32,128],[kernelSizeBlocks 1]);
[lgraph,lastStage] = addIdentityBlock2D(lgraph,lastStage,...
    'idBlock3d',[32,32,128],[kernelSizeBlocks 1]);

%% 5) Stage 4: convBlock(stride=2) + 5 identityBlocks
[lgraph,lastStage] = addConvBlock2D(lgraph,lastStage,...
    'convBlock4a',[64,64,256],[kernelSizeBlocks 1],'Stride',[2 1]);

[lgraph,lastStage] = addIdentityBlock2D(lgraph,lastStage,...
    'idBlock4b',[64,64,256],[kernelSizeBlocks 1]);
[lgraph,lastStage] = addIdentityBlock2D(lgraph,lastStage,...
    'idBlock4c',[64,64,256],[kernelSizeBlocks 1]);
[lgraph,lastStage] = addIdentityBlock2D(lgraph,lastStage,...
    'idBlock4d',[64,64,256],[kernelSizeBlocks 1]);
[lgraph,lastStage] = addIdentityBlock2D(lgraph,lastStage,...
    'idBlock4e',[64,64,256],[kernelSizeBlocks 1]);
[lgraph,lastStage] = addIdentityBlock2D(lgraph,lastStage,...
    'idBlock4f',[64,64,256],[kernelSizeBlocks 1]);

%% 6) Stage 5: convBlock(stride=2) + 2 identityBlocks
[lgraph,lastStage] = addConvBlock2D(lgraph,lastStage,...
    'convBlock5a',[64,64,256],[kernelSizeBlocks 1],'Stride',[2 1]);

[lgraph,lastStage] = addIdentityBlock2D(lgraph,lastStage,...
    'idBlock5b',[64,64,256],[kernelSizeBlocks 1]);
[lgraph,lastStage] = addIdentityBlock2D(lgraph,lastStage,...
    'idBlock5c',[64,64,256],[kernelSizeBlocks 1]);

%% 7) Final: averagePooling2dLayer -> flattenLayer -> fullyConnectedLayer(nOut)
%   (Optionally add a 'regressionLayer' if needed)
avgPool = averagePooling2dLayer([5 1], 'Name','avgPool',...
    'Stride',[5 1]); 
flatten = flattenLayer("Name","flatten");
fcOut   = fullyConnectedLayer(nOut,'Name','fc-out');



lgraph = addLayers(lgraph,avgPool);
lgraph = addLayers(lgraph,flatten);
lgraph = addLayers(lgraph,fcOut);


% Connect them
lgraph = connectLayers(lgraph,lastStage,'avgPool');
lgraph = connectLayers(lgraph,'avgPool','flatten');
lgraph = connectLayers(lgraph,'flatten','fc-out');

% (Now analyzeNetwork(lgraph) or return it.)
end


function [lgraph, outName] = addConvBlock2D(lgraph, inName, prefix, ...
    filters, kernelSize2D, varargin)
% addConvBlock2D - "conv_block" for a 2D ResNet block:
%    3-layer main path + 1-layer shortcut => add => ReLU
%
% Usage example:
%   [lgraph,outName] = addConvBlock2D(lgraph, 'pool1','convBlock2a',...
%       [16,16,64], [3 1], 'Stride',[2 1]);

p = inputParser();
addParameter(p,'Stride',[2 1],@(x) isnumeric(x) && numel(x)==2);
parse(p,varargin{:});
strideVal = p.Results.Stride;

[f1,f2,f3] = deal(filters(1), filters(2), filters(3));

% MAIN path
% 1) Conv A
convA = convolution2dLayer([1 1], f1,...
    'Name',[prefix '_2a'],...
    'Stride', strideVal,...
    'Padding','same');
bnA   = batchNormalizationLayer('Name',[prefix '_bn2a']);
reluA = reluLayer('Name',[prefix '_relu2a']);

% 2) Conv B
convB = convolution2dLayer(kernelSize2D, f2, ...
    'Name',[prefix '_2b'],...
    'Padding','same');
bnB   = batchNormalizationLayer('Name',[prefix '_bn2b']);
reluB = reluLayer('Name',[prefix '_relu2b']);

% 3) Conv C
convC = convolution2dLayer([1 1], f3,...
    'Name',[prefix '_2c'],...
    'Padding','same');
bnC   = batchNormalizationLayer('Name',[prefix '_bn2c']);

% SHORTCUT path
convS = convolution2dLayer([1 1], f3, ...
    'Name',[prefix '_1'],...
    'Stride',strideVal,...
    'Padding','same');
bnS   = batchNormalizationLayer('Name',[prefix '_bn1']);

% ADD + final ReLU
addL = additionLayer(2,'Name',[prefix '_add']);
reluF= reluLayer('Name',[prefix '_out']);

% Add each layer individually
lgraph = addLayers(lgraph,convA);
lgraph = addLayers(lgraph,bnA);
lgraph = addLayers(lgraph,reluA);

lgraph = addLayers(lgraph,convB);
lgraph = addLayers(lgraph,bnB);
lgraph = addLayers(lgraph,reluB);

lgraph = addLayers(lgraph,convC);
lgraph = addLayers(lgraph,bnC);

lgraph = addLayers(lgraph,convS);
lgraph = addLayers(lgraph,bnS);

lgraph = addLayers(lgraph,addL);
lgraph = addLayers(lgraph,reluF);

% Connect main path
lgraph = connectLayers(lgraph,inName,[prefix '_2a']);
lgraph = connectLayers(lgraph,[prefix '_2a'],[prefix '_bn2a']);
lgraph = connectLayers(lgraph,[prefix '_bn2a'],[prefix '_relu2a']);
lgraph = connectLayers(lgraph,[prefix '_relu2a'],[prefix '_2b']);
lgraph = connectLayers(lgraph,[prefix '_2b'],[prefix '_bn2b']);
lgraph = connectLayers(lgraph,[prefix '_bn2b'],[prefix '_relu2b']);
lgraph = connectLayers(lgraph,[prefix '_relu2b'],[prefix '_2c']);
lgraph = connectLayers(lgraph,[prefix '_2c'],[prefix '_bn2c']);

% Connect shortcut path
lgraph = connectLayers(lgraph,inName,[prefix '_1']);
lgraph = connectLayers(lgraph,[prefix '_1'],[prefix '_bn1']);

% add => final ReLU
lgraph = connectLayers(lgraph,[prefix '_bn2c'], [prefix '_add' '/in1']);
lgraph = connectLayers(lgraph,[prefix '_bn1'],  [prefix '_add' '/in2']);
lgraph = connectLayers(lgraph,[prefix '_add'],  [prefix '_out']);

outName = [prefix '_out'];
end


function [lgraph, outName] = addIdentityBlock2D(lgraph, inName, ...
    prefix, filters, kernelSize2D, varargin)
% addIdentityBlock2D - "no stride" 2D block: 3-layer main path + direct skip => add => ReLU
%
% Usage:
%   [lgraph,outName] = addIdentityBlock2D(lgraph,'pool1','idBlock2b',...
%       [16,16,64],[3 1]);

p = inputParser();
addParameter(p,'Stride',[1 1],@(x) isnumeric(x)&&numel(x)==2);
parse(p,varargin{:});
% strideVal = p.Results.Stride; % Not typically used in identity blocks

[f1,f2,f3] = deal(filters(1), filters(2), filters(3));

% MAIN path
convA = convolution2dLayer([1 1], f1,...
    'Name',[prefix '_2a'],'Padding','same');
bnA   = batchNormalizationLayer('Name',[prefix '_bn2a']);
reluA = reluLayer('Name',[prefix '_relu2a']);

convB = convolution2dLayer(kernelSize2D, f2,...
    'Name',[prefix '_2b'],'Padding','same');
bnB   = batchNormalizationLayer('Name',[prefix '_bn2b']);
reluB = reluLayer('Name',[prefix '_relu2b']);

convC = convolution2dLayer([1 1], f3,...
    'Name',[prefix '_2c'],'Padding','same');
bnC   = batchNormalizationLayer('Name',[prefix '_bn2c']);

% ADD + final ReLU
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

% Connect main path
lgraph = connectLayers(lgraph, inName,[prefix '_2a']);
lgraph = connectLayers(lgraph, [prefix '_2a'], [prefix '_bn2a']);
lgraph = connectLayers(lgraph, [prefix '_bn2a'], [prefix '_relu2a']);
lgraph = connectLayers(lgraph, [prefix '_relu2a'], [prefix '_2b']);
lgraph = connectLayers(lgraph, [prefix '_2b'], [prefix '_bn2b']);
lgraph = connectLayers(lgraph, [prefix '_bn2b'], [prefix '_relu2b']);
lgraph = connectLayers(lgraph, [prefix '_relu2b'], [prefix '_2c']);
lgraph = connectLayers(lgraph, [prefix '_2c'], [prefix '_bn2c']);

% Direct skip path => the other input to addition
lgraph = connectLayers(lgraph, inName, [prefix '_add' '/in2']);
lgraph = connectLayers(lgraph,[prefix '_bn2c'], [prefix '_add' '/in1']);
lgraph = connectLayers(lgraph,[prefix '_add'], [prefix '_out']);

outName = [prefix '_out'];
end
