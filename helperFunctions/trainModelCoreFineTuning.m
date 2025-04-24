function trainedNet = trainModelCoreFineTuning(oldModel, Xtrain, Ytrain)
% trainModelCoreFineTuning Fine tunes an existing network starting from oldModel.
%
%   trainedNet = trainModelCoreFineTuning(oldModel, Xtrain, Ytrain)
%
%   Inputs:
%       oldModel - Pretrained dlnetwork to be fine tuned.
%       Xtrain   - Training input data, with dimensions [C x T x N].
%       Ytrain   - Training labels, [N x 1].
%
%   This function shuffles the training data, prepares the dlarray, and
%   then fine tunes the network using trainingOptions with a lower learning rate.
%
%   Note: Adjust hyperparameters (e.g., MaxEpochs, MiniBatchSize, InitialLearnRate)
%         as needed for your fine-tuning task.

nTrain = size(Xtrain, 3);
fprintf('    Fine tuning with %d curves.\n', nTrain);

% Use the provided oldModel as the starting network.
net = oldModel;

% Permute input: [numFeatures x numSamples x sequenceLength] expected by training
Xperm = permute(Xtrain, [1, 3, 2]);  % [C x N x T]

% Shuffle the data
rp = randperm(nTrain);
Xperm = Xperm(:, rp, :);
Ytrain = Ytrain(rp);

dlX = dlarray(Xperm, 'CBT');  % 'C' = channels, 'B' = batch, 'T' = sequence length

opts = trainingOptions('adam',...
    'MaxEpochs',100,...             % Fewer epochs for fine tuning
    'MiniBatchSize',32,...          % Adjust batch size as needed
    'Shuffle','every-epoch',...
    'Verbose',true,...
    'Plots','none',...
    'ValidationFrequency',50,...
    'InitialLearnRate',1e-4, ...    % Lower learning rate for fine tuning
    'L2Regularization', 1e-4, ...
    'Epsilon',1e-8);

try
    [trainedNet, info] = trainnet(dlX, Ytrain, net, "mae", opts);
catch ME
    warning('Error in trainModelCoreFineTuning:\n%s', ME.message);
    trainedNet = net; % Return the original network in case of error.
end

end
