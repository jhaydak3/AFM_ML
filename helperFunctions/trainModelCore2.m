function trainedNet = trainModelCore2(lgraph, Xtrain, Ytrain)
% trainModelCore2  Train a CNN using "image" format for 1D data.
%
%   * Xtrain is assumed to be [C x T x B] = [numFeatures x seqLen x numSamples]
%   * We permute/reshape to [seqLen x 1 x numFeatures x batchSize] so that
%     it matches the imageInputLayer shape [H x W x C x N].
%   * Ytrain should be [1 x B] for single-scalar regression per sample.

nTrain = size(Xtrain, 3);
fprintf('    Actually training with %d samples.\n', nTrain);

% (1) Reformat data from [C x T x B] to [T x 1 x C x B].
%     - Dimension 1 (C) -> dimension 3
%     - Dimension 2 (T) -> dimension 1
%     - Dimension 3 (B) -> dimension 4
Xperm = permute(Xtrain, [2 4 1 3]);
% Now Xperm is [T x ? x C x B]. The "?" dimension is missing, so let's insert W=1:
% Actually easiest is just: [T x C x B] => [T x 1 x C x B]
Ximgs = reshape(Xperm, [size(Xperm,1), 1, size(Xperm,3), size(Xperm,4)]);

% (2) Ensure Ytrain is shape [1 x B], so each sample is one scalar label.
if iscolumn(Ytrain)
    Ytrain = Ytrain';  % now [1 x B]
end

% (3) Setup training options
opts = trainingOptions('adam', ...
    'MaxEpochs', 200, ...
    'MiniBatchSize', 64*.5, ...
    'Shuffle','every-epoch', ...
    'Verbose', true, ...
    'Plots', 'none', ...
    'ValidationFrequency', 50, ...
    'InitialLearnRate', 1e-4, ...
    'L2Regularization', 1e-4, ...
    'Epsilon', 1e-8);

% (4) Train the network
trainedNet = trainnet(Ximgs, Ytrain', dlnetwork(lgraph),'mae', opts);

end
