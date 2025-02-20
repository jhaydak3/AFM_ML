function trainedNet = trainModelCore(layers, Xtrain, Ytrain)
% This is the "core" function to actually train given Xtrain, Ytrain
%  - For classification: "crossentropy"
%  - For regression: "mae" or "mse"
%
% 
%

nTrain = size(Xtrain,3);
fprintf('    Actually training with %d curves.\n', nTrain);

net = dlnetwork(layers);

Xperm = permute(Xtrain, [1,3,2]); % [C x B x T]

% Shuffle
rp = randperm(nTrain);
Xperm = Xperm(:, rp, :);
Ytrain = Ytrain(rp);

dlX   = dlarray(Xperm, 'CBT');




opts = trainingOptions('adam',...
    'MaxEpochs',200,...
    'MiniBatchSize',64*.5,...
    'Shuffle','every-epoch',...
    'Verbose',true,...
    'Plots','none',...
    'ValidationFrequency',50,...
    'InitialLearnRate',1e-4, ...
    'L2Regularization', 1e-4, ...
    'Epsilon',1e-8);

try
    % NO transpose of Ytrain here (assuming classification with B= #samples)
    [trainedNet, info] = trainnet(dlX, Ytrain, net, "mae", opts);
catch ME
    warning('Error in trainModelCore.\n%s', ME.message);

end
end
