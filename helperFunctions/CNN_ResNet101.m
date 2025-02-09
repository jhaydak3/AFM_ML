function lgraph = CNN_ResNet101(numFeatures,sequenceLength)
% resnet101_1D_for_regression
% Creates a complete layerGraph that implements a 1D ResNet-101 for
% regression (scalar output).
%
% Architecture in detail:
%   [STEM]        : conv(7,64,stride=2) -> bn -> relu -> maxpool(3,stride=2)
%   [res2] group  : 3 blocks (res2a, res2b, res2c)
%   [res3] group  : 4 blocks (res3a, res3b1, res3b2, res3b3)
%   [res4] group  : 23 blocks (res4a, res4b1 ... res4b22)
%   [res5] group  : 3 blocks (res5a, res5b, res5c)
%   [head]        : globalAvgPool1d -> fc(1) -> regression
%
% No pretrained parameters or bias/weight constraints are included.
% All conv and BN layers will be randomly initialized at training time.

%--------------------------------------------------------------------------
% 1) Create an empty layer graph.
%--------------------------------------------------------------------------
lgraph = layerGraph();

%--------------------------------------------------------------------------
% 2) Add the "stem": sequenceInput -> conv1 -> bn -> relu -> maxpool
%--------------------------------------------------------------------------
stemLayers = [
    sequenceInputLayer(numFeatures,"Name","data",'MinLength',sequenceLength)
    convolution1dLayer(3,64,"Name","conv1", ...
        "Padding","same","Stride",2)
    batchNormalizationLayer("Name","bn_conv1")
    reluLayer("Name","conv1_relu")
    maxPooling1dLayer(2,"Name","pool1","Padding","same","Stride",2)
];
lgraph = addLayers(lgraph, stemLayers);

%--------------------------------------------------------------------------
% 3) Add res2 group (3 blocks: res2a, res2b, res2c)
%     - res2a has a "branch2" and a "branch1" (skip) with no stride (1).
%     - res2b, res2c each have only branch2, plus addition + relu.
%--------------------------------------------------------------------------

%----- res2a branch2
res2a_branch2 = [
    convolution1dLayer(1,64,"Name","res2a_branch2a","Padding","same")
    batchNormalizationLayer("Name","bn2a_branch2a")
    reluLayer("Name","res2a_branch2a_relu")

    convolution1dLayer(3,64,"Name","res2a_branch2b","Padding","same")
    batchNormalizationLayer("Name","bn2a_branch2b")
    reluLayer("Name","res2a_branch2b_relu")

    convolution1dLayer(1,256,"Name","res2a_branch2c","Padding","same")
    batchNormalizationLayer("Name","bn2a_branch2c")
];
lgraph = addLayers(lgraph, res2a_branch2);

%----- res2a branch1 (skip)
res2a_branch1 = [
    convolution1dLayer(1,256,"Name","res2a_branch1","Padding","same")
    batchNormalizationLayer("Name","bn2a_branch1")
];
lgraph = addLayers(lgraph, res2a_branch1);

%----- res2a addition + relu
res2a_addrelu = [
    additionLayer(2,"Name","res2a")
    reluLayer("Name","res2a_relu")
];
lgraph = addLayers(lgraph, res2a_addrelu);

%----- res2b block
res2b_branch2 = [
    convolution1dLayer(1,64,"Name","res2b_branch2a","Padding","same")
    batchNormalizationLayer("Name","bn2b_branch2a")
    reluLayer("Name","res2b_branch2a_relu")

    convolution1dLayer(3,64,"Name","res2b_branch2b","Padding","same")
    batchNormalizationLayer("Name","bn2b_branch2b")
    reluLayer("Name","res2b_branch2b_relu")

    convolution1dLayer(1,256,"Name","res2b_branch2c","Padding","same")
    batchNormalizationLayer("Name","bn2b_branch2c")
];
lgraph = addLayers(lgraph, res2b_branch2);

res2b_addrelu = [
    additionLayer(2,"Name","res2b")
    reluLayer("Name","res2b_relu")
];
lgraph = addLayers(lgraph, res2b_addrelu);

%----- res2c block
res2c_branch2 = [
    convolution1dLayer(1,64,"Name","res2c_branch2a","Padding","same")
    batchNormalizationLayer("Name","bn2c_branch2a")
    reluLayer("Name","res2c_branch2a_relu")

    convolution1dLayer(3,64,"Name","res2c_branch2b","Padding","same")
    batchNormalizationLayer("Name","bn2c_branch2b")
    reluLayer("Name","res2c_branch2b_relu")

    convolution1dLayer(1,256,"Name","res2c_branch2c","Padding","same")
    batchNormalizationLayer("Name","bn2c_branch2c")
];
lgraph = addLayers(lgraph, res2c_branch2);

res2c_addrelu = [
    additionLayer(2,"Name","res2c")
    reluLayer("Name","res2c_relu")
];
lgraph = addLayers(lgraph, res2c_addrelu);

%--------------------------------------------------------------------------
% 4) Add res3 group (4 blocks: res3a, res3b1, res3b2, res3b3)
%     - res3a has stride=2 on branch2a and branch1
%--------------------------------------------------------------------------

%----- res3a branch2
res3a_branch2 = [
    convolution1dLayer(1,128,"Name","res3a_branch2a","Stride",2,"Padding","same")
    batchNormalizationLayer("Name","bn3a_branch2a")
    reluLayer("Name","res3a_branch2a_relu")

    convolution1dLayer(3,128,"Name","res3a_branch2b","Padding","same")
    batchNormalizationLayer("Name","bn3a_branch2b")
    reluLayer("Name","res3a_branch2b_relu")

    convolution1dLayer(1,512,"Name","res3a_branch2c","Padding","same")
    batchNormalizationLayer("Name","bn3a_branch2c")
];
lgraph = addLayers(lgraph, res3a_branch2);

%----- res3a branch1
res3a_branch1 = [
    convolution1dLayer(1,512,"Name","res3a_branch1","Stride",2,"Padding","same")
    batchNormalizationLayer("Name","bn3a_branch1")
];
lgraph = addLayers(lgraph, res3a_branch1);

res3a_addrelu = [
    additionLayer(2,"Name","res3a")
    reluLayer("Name","res3a_relu")
];
lgraph = addLayers(lgraph, res3a_addrelu);

%----- res3b1
res3b1_branch2 = [
    convolution1dLayer(1,128,"Name","res3b1_branch2a","Padding","same")
    batchNormalizationLayer("Name","bn3b1_branch2a")
    reluLayer("Name","res3b1_branch2a_relu")

    convolution1dLayer(3,128,"Name","res3b1_branch2b","Padding","same")
    batchNormalizationLayer("Name","bn3b1_branch2b")
    reluLayer("Name","res3b1_branch2b_relu")

    convolution1dLayer(1,512,"Name","res3b1_branch2c","Padding","same")
    batchNormalizationLayer("Name","bn3b1_branch2c")
];
lgraph = addLayers(lgraph, res3b1_branch2);

res3b1_addrelu = [
    additionLayer(2,"Name","res3b1")
    reluLayer("Name","res3b1_relu")
];
lgraph = addLayers(lgraph, res3b1_addrelu);

%----- res3b2
res3b2_branch2 = [
    convolution1dLayer(1,128,"Name","res3b2_branch2a","Padding","same")
    batchNormalizationLayer("Name","bn3b2_branch2a")
    reluLayer("Name","res3b2_branch2a_relu")

    convolution1dLayer(3,128,"Name","res3b2_branch2b","Padding","same")
    batchNormalizationLayer("Name","bn3b2_branch2b")
    reluLayer("Name","res3b2_branch2b_relu")

    convolution1dLayer(1,512,"Name","res3b2_branch2c","Padding","same")
    batchNormalizationLayer("Name","bn3b2_branch2c")
];
lgraph = addLayers(lgraph, res3b2_branch2);

res3b2_addrelu = [
    additionLayer(2,"Name","res3b2")
    reluLayer("Name","res3b2_relu")
];
lgraph = addLayers(lgraph, res3b2_addrelu);

%----- res3b3
res3b3_branch2 = [
    convolution1dLayer(1,128,"Name","res3b3_branch2a","Padding","same")
    batchNormalizationLayer("Name","bn3b3_branch2a")
    reluLayer("Name","res3b3_branch2a_relu")

    convolution1dLayer(3,128,"Name","res3b3_branch2b","Padding","same")
    batchNormalizationLayer("Name","bn3b3_branch2b")
    reluLayer("Name","res3b3_branch2b_relu")

    convolution1dLayer(1,512,"Name","res3b3_branch2c","Padding","same")
    batchNormalizationLayer("Name","bn3b3_branch2c")
];
lgraph = addLayers(lgraph, res3b3_branch2);

res3b3_addrelu = [
    additionLayer(2,"Name","res3b3")
    reluLayer("Name","res3b3_relu")
];
lgraph = addLayers(lgraph, res3b3_addrelu);

%--------------------------------------------------------------------------
% 5) Add res4 group (23 blocks: res4a, res4b1 ... res4b22)
%     - res4a has stride=2 on branch2a and branch1
%     - each res4bX has the same shape as normal: (1->3->1) conv
%--------------------------------------------------------------------------

%----- res4a
res4a_branch2 = [
    convolution1dLayer(1,256,"Name","res4a_branch2a","Stride",2,"Padding","same")
    batchNormalizationLayer("Name","bn4a_branch2a")
    reluLayer("Name","res4a_branch2a_relu")

    convolution1dLayer(3,256,"Name","res4a_branch2b","Padding","same")
    batchNormalizationLayer("Name","bn4a_branch2b")
    reluLayer("Name","res4a_branch2b_relu")

    convolution1dLayer(1,1024,"Name","res4a_branch2c","Padding","same")
    batchNormalizationLayer("Name","bn4a_branch2c")
];
lgraph = addLayers(lgraph, res4a_branch2);

res4a_branch1 = [
    convolution1dLayer(1,1024,"Name","res4a_branch1","Stride",2,"Padding","same")
    batchNormalizationLayer("Name","bn4a_branch1")
];
lgraph = addLayers(lgraph, res4a_branch1);

res4a_addrelu = [
    additionLayer(2,"Name","res4a")
    reluLayer("Name","res4a_relu")
];
lgraph = addLayers(lgraph, res4a_addrelu);

% We now add res4b1 through res4b22. Each block has:
%   branch2: conv(1,256) -> bn -> relu -> conv(3,256)-> bn-> relu -> conv(1,1024)-> bn
%   additionLayer -> reluLayer
%
% Let’s define a little helper to build each block quickly:
%
% blockName is like "res4b1"
% midFilters = 256, outFilters = 1024
% (stride=1, because only res4a had stride=2)

for bIdx = 1:22
    % e.g. "res4b1", "res4b2", ...
    blockName = sprintf("res4b%d", bIdx);
    blockName = char(blockName);

    % Branch2
    branch2Layers = [
        convolution1dLayer(1,256, ...
            "Name",[blockName, '_branch2a'],"Padding","same")
        batchNormalizationLayer("Name",['bn4b', num2str(bIdx), '_branch2a'])
        reluLayer("Name",[blockName, '_branch2a_relu'])
        
        convolution1dLayer(3,256, ...
            "Name",[blockName, '_branch2b'],"Padding","same")
        batchNormalizationLayer("Name",['bn4b', num2str(bIdx), '_branch2b'])
        reluLayer("Name",[blockName, '_branch2b_relu'])
        
        convolution1dLayer(1,1024, ...
            "Name",[blockName, '_branch2c'],"Padding","same")
        batchNormalizationLayer("Name",['bn4b', num2str(bIdx), '_branch2c'])
    ];
    lgraph = addLayers(lgraph, branch2Layers);

    % addition + relu
    addReluLayers = [
        additionLayer(2,"Name",blockName)
        reluLayer("Name",[blockName, '_relu'])
    ];
    lgraph = addLayers(lgraph, addReluLayers);
end

%--------------------------------------------------------------------------
% 6) Add res5 group (3 blocks: res5a, res5b, res5c)
%     - res5a has stride=2 on branch2a and branch1
%--------------------------------------------------------------------------
%----- res5a
res5a_branch2 = [
    convolution1dLayer(1,512,"Name","res5a_branch2a","Stride",2,"Padding","same")
    batchNormalizationLayer("Name","bn5a_branch2a")
    reluLayer("Name","res5a_branch2a_relu")

    convolution1dLayer(3,512,"Name","res5a_branch2b","Padding","same")
    batchNormalizationLayer("Name","bn5a_branch2b")
    reluLayer("Name","res5a_branch2b_relu")

    convolution1dLayer(1,2048,"Name","res5a_branch2c","Padding","same")
    batchNormalizationLayer("Name","bn5a_branch2c")
];
lgraph = addLayers(lgraph, res5a_branch2);

res5a_branch1 = [
    convolution1dLayer(1,2048,"Name","res5a_branch1","Stride",2,"Padding","same")
    batchNormalizationLayer("Name","bn5a_branch1")
];
lgraph = addLayers(lgraph, res5a_branch1);

res5a_addrelu = [
    additionLayer(2,"Name","res5a")
    reluLayer("Name","res5a_relu")
];
lgraph = addLayers(lgraph, res5a_addrelu);

%----- res5b
res5b_branch2 = [
    convolution1dLayer(1,512,"Name","res5b_branch2a","Padding","same")
    batchNormalizationLayer("Name","bn5b_branch2a")
    reluLayer("Name","res5b_branch2a_relu")

    convolution1dLayer(3,512,"Name","res5b_branch2b","Padding","same")
    batchNormalizationLayer("Name","bn5b_branch2b")
    reluLayer("Name","res5b_branch2b_relu")

    convolution1dLayer(1,2048,"Name","res5b_branch2c","Padding","same")
    batchNormalizationLayer("Name","bn5b_branch2c")
];
lgraph = addLayers(lgraph, res5b_branch2);

res5b_addrelu = [
    additionLayer(2,"Name","res5b")
    reluLayer("Name","res5b_relu")
];
lgraph = addLayers(lgraph, res5b_addrelu);

%----- res5c
res5c_branch2 = [
    convolution1dLayer(1,512,"Name","res5c_branch2a","Padding","same")
    batchNormalizationLayer("Name","bn5c_branch2a")
    reluLayer("Name","res5c_branch2a_relu")

    convolution1dLayer(3,512,"Name","res5c_branch2b","Padding","same")
    batchNormalizationLayer("Name","bn5c_branch2b")
    reluLayer("Name","res5c_branch2b_relu")

    convolution1dLayer(1,2048,"Name","res5c_branch2c","Padding","same")
    batchNormalizationLayer("Name","bn5c_branch2c")
];
lgraph = addLayers(lgraph, res5c_branch2);

res5c_addrelu_head = [
    additionLayer(2,"Name","res5c")
    reluLayer("Name","res5c_relu")
    globalAveragePooling1dLayer("Name","pool5")
    fullyConnectedLayer(1,"Name","fc")        % single scalar output
    %regressionLayer("Name","regressionoutput")
];
lgraph = addLayers(lgraph, res5c_addrelu_head);

%--------------------------------------------------------------------------
% 7) Now connect everything
%--------------------------------------------------------------------------

%
% ---- STEM to res2a
%
lgraph = connectLayers(lgraph,"pool1","res2a_branch2a");
lgraph = connectLayers(lgraph,"pool1","res2a_branch1");
lgraph = connectLayers(lgraph,"bn2a_branch2c","res2a/in1");
lgraph = connectLayers(lgraph,"bn2a_branch1","res2a/in2");

%
% ---- res2a -> res2b
%
lgraph = connectLayers(lgraph,"res2a_relu","res2b_branch2a");
lgraph = connectLayers(lgraph,"res2a_relu","res2b/in2");
lgraph = connectLayers(lgraph,"bn2b_branch2c","res2b/in1");

%
% ---- res2b -> res2c
%
lgraph = connectLayers(lgraph,"res2b_relu","res2c_branch2a");
lgraph = connectLayers(lgraph,"res2b_relu","res2c/in2");
lgraph = connectLayers(lgraph,"bn2c_branch2c","res2c/in1");

%
% ---- res2c -> res3a
%
lgraph = connectLayers(lgraph,"res2c_relu","res3a_branch2a");
lgraph = connectLayers(lgraph,"res2c_relu","res3a_branch1");
lgraph = connectLayers(lgraph,"bn3a_branch2c","res3a/in1");
lgraph = connectLayers(lgraph,"bn3a_branch1","res3a/in2");

%
% ---- res3a -> res3b1
%
lgraph = connectLayers(lgraph,"res3a_relu","res3b1_branch2a");
lgraph = connectLayers(lgraph,"res3a_relu","res3b1/in2");
lgraph = connectLayers(lgraph,"bn3b1_branch2c","res3b1/in1");

%
% ---- res3b1 -> res3b2
%
lgraph = connectLayers(lgraph,"res3b1_relu","res3b2_branch2a");
lgraph = connectLayers(lgraph,"res3b1_relu","res3b2/in2");
lgraph = connectLayers(lgraph,"bn3b2_branch2c","res3b2/in1");

%
% ---- res3b2 -> res3b3
%
lgraph = connectLayers(lgraph,"res3b2_relu","res3b3_branch2a");
lgraph = connectLayers(lgraph,"res3b2_relu","res3b3/in2");
lgraph = connectLayers(lgraph,"bn3b3_branch2c","res3b3/in1");

%
% ---- res3b3 -> res4a
%
lgraph = connectLayers(lgraph,"res3b3_relu","res4a_branch2a");
lgraph = connectLayers(lgraph,"res3b3_relu","res4a_branch1");
lgraph = connectLayers(lgraph,"bn4a_branch2c","res4a/in1");
lgraph = connectLayers(lgraph,"bn4a_branch1","res4a/in2");

%
% ---- res4a -> res4b1
%
lgraph = connectLayers(lgraph,"res4a_relu","res4b1_branch2a");
lgraph = connectLayers(lgraph,"res4a_relu","res4b1/in2");
%lgraph = connectLayers(lgraph,"bn4b1_branch2c","res4b1/in1");

% We must connect each subsequent res4bX in a chain:
%  res4b1 -> res4b2 -> res4b3 -> ... -> res4b22

for bIdx = 1:22
    currBlockName = sprintf("res4b%d", bIdx);
    currBlockName = char(currBlockName);
    nextBlockIdx  = bIdx + 1; 
    nextBlockName = sprintf("res4b%d", nextBlockIdx);

    % Connect the branch2’s final BN to the addition layer:
    lgraph = connectLayers(lgraph, ...
        ['bn4b' num2str(bIdx) '_branch2c'], [currBlockName  '/in1']);

    % If bIdx < 22, connect the ReLU output to next block’s branch2 and skip:
    if bIdx < 22
        lgraph = connectLayers(lgraph, ...
            currBlockName + "_relu", nextBlockName + "_branch2a");
        lgraph = connectLayers(lgraph, ...
            currBlockName + "_relu", nextBlockName + "/in2");
    end
end

% Now, “res4b22_relu” should connect to res5a (the next group):
lgraph = connectLayers(lgraph,"res4b22_relu","res5a_branch2a");
lgraph = connectLayers(lgraph,"res4b22_relu","res5a_branch1");
lgraph = connectLayers(lgraph,"bn5a_branch2c","res5a/in1");
lgraph = connectLayers(lgraph,"bn5a_branch1","res5a/in2");

%
% ---- res5a -> res5b
%
lgraph = connectLayers(lgraph,"res5a_relu","res5b_branch2a");
lgraph = connectLayers(lgraph,"res5a_relu","res5b/in2");
lgraph = connectLayers(lgraph,"bn5b_branch2c","res5b/in1");

%
% ---- res5b -> res5c
%
lgraph = connectLayers(lgraph,"res5b_relu","res5c_branch2a");
lgraph = connectLayers(lgraph,"res5b_relu","res5c/in2");
lgraph = connectLayers(lgraph,"bn5c_branch2c","res5c/in1");

%--------------------------------------------------------------------------
% Done!
%--------------------------------------------------------------------------

end
