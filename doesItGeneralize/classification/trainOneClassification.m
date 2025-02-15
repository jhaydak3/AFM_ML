function trainedStruct = trainOneClassification(layers, ...
    preProcessedDataFileTrain, numSamplesToTrain, saveTrainedModel, saveModelFileName)
% TRAINONECLASSIFICATION
%
% Trains up to TWO models if dataset has < numSamplesToTrain:
%   1) modelForOthers: trained on 100% of the data
%   2) modelForSelf:   trained on 80% of the data, leaving 20% for internal test
%
% If dataset has >= numSamplesToTrain, trains only ONE model (the usual):
%   modelForOthers = same as "big" model using exactly numSamplesToTrain
%   (no modelForSelf is created)
%
% Returns a struct with fields:
%   .modelForOthers      => dlnetwork for cross-dataset testing
%   .modelForSelf        => dlnetwork for self-dataset testing (may be [])
%   .selfTestIndices     => indices of the leftover 20% (if .modelForSelf is used)
%   .trainIndicesUsed    => which indices are used for "modelForOthers"
%
% Inputs:
%   layers                  : your CNN layers
%   preProcessedDataFileTrain: .mat file with X, goodOrBad
%   numSamplesToTrain       : threshold (e.g. 500)
%   saveTrainedModel        : bool
%   saveModelFileName       : path to .mat file to save
%
% Output:
%   trainedStruct           : struct with up to two models

    rng(1337,'twister');  % reproducible

    fprintf('Loading classification data from: %s\n', preProcessedDataFileTrain);
    dataTrain = load(preProcessedDataFileTrain);

    % Basic checks
    if ~isfield(dataTrain,'X') || ~isfield(dataTrain,'goodOrBad')
        error('Data file "%s" missing X or goodOrBad.', preProcessedDataFileTrain);
    end
    XFull = dataTrain.X;          % [features x seqLength x N]
    labelsFull = dataTrain.goodOrBad;  % 0=reject, 1=accept

    N = size(XFull,3);
    fprintf('Dataset has %d total samples.\n', N);

    trainedStruct = struct();
    trainedStruct.modelForOthers  = [];
    trainedStruct.modelForSelf    = [];
    trainedStruct.selfTestIndices = [];
    trainedStruct.trainIndicesUsed= [];

    if N < 1
        warning('No data to train on at all => returning empty struct.');
        return;
    end

    % Always create "modelForOthers" 
    %  - If N >= numSamplesToTrain => pick exactly numSamplesToTrain
    %  - Else => use 100% of data
    if N >= numSamplesToTrain
        % pick exactly numSamplesToTrain
        permAll = randperm(N);
        trainInd  = permAll(1:numSamplesToTrain);
        trainedStruct.trainIndicesUsed = trainInd;

        Xtrain = XFull(:,:, trainInd);
        Ytrain = labelsFull(trainInd);
        fprintf('Using %d samples for modelForOthers.\n', numSamplesToTrain);

    else
        % use ALL data
        trainInd  = 1:N;  % entire set
        trainedStruct.trainIndicesUsed = trainInd;

        Xtrain = XFull;
        Ytrain = labelsFull;
        fprintf('Data < numSamplesToTrain => using ALL %d for modelForOthers.\n', N);
    end

    % Convert labels => categorical
    strTrain = strings(length(trainInd),1);
    strTrain(Ytrain==0) = "reject";
    strTrain(Ytrain==1) = "accept";
    YtrainCat = categorical(strTrain, {'reject','accept'});

    % Build the net
    net1 = dlnetwork(layers);

    % Convert to 'CBT'
    XtrainPerm = permute(Xtrain, [1,3,2]);
    dlXtrain   = dlarray(XtrainPerm, 'CBT');

    % Shuffle
    rp = randperm(length(trainInd));
    dlXtrain   = dlXtrain(:,rp,:);
    YtrainCat  = YtrainCat(rp);

    options = trainingOptions('adam',...
        'MaxEpochs',30,...
        'MiniBatchSize',64,...
        'Shuffle','every-epoch',...
        'Verbose',true,...
        'Plots','none',...
        'ValidationFrequency',50,...
        'InitialLearnRate',5e-4);

    fprintf('Training modelForOthers...\n');
    [modelForOthers, info1] = trainnet(dlXtrain, YtrainCat, net1, "crossentropy", options);
    trainedStruct.modelForOthers = modelForOthers;

    % If N < numSamplesToTrain => we also create "modelForSelf" with 80/20 split
    if N < numSamplesToTrain
        fprintf('Dataset < %d => creating modelForSelf w/ 80/20 split...\n', numSamplesToTrain);
        rp2 = randperm(N);
        numTrain2 = max(1, floor(0.8 * N));
        trainInd2 = rp2(1:numTrain2);
        testInd2  = rp2(numTrain2+1:end);

        trainedStruct.selfTestIndices = testInd2;  % leftover 20%

        Xtrain2 = XFull(:,:, trainInd2);
        Ytrain2 = labelsFull(trainInd2);

        strTrain2 = strings(length(trainInd2),1);
        strTrain2(Ytrain2==0)="reject";
        strTrain2(Ytrain2==1)="accept";
        Ytrain2Cat = categorical(strTrain2, {'reject','accept'});

        net2 = dlnetwork(layers);

        Xtrain2Perm = permute(Xtrain2, [1,3,2]);
        dlXtrain2    = dlarray(Xtrain2Perm, 'CBT');
        rp3 = randperm(length(trainInd2));
        dlXtrain2  = dlXtrain2(:,rp3,:);
        Ytrain2Cat = Ytrain2Cat(rp3);

        fprintf('Training modelForSelf on ~80%% of data...\n');
        [modelForSelf, info2] = trainnet(dlXtrain2, Ytrain2Cat, net2, "crossentropy", options);

        trainedStruct.modelForSelf = modelForSelf;
    end

    % (Optional) Save
    if saveTrainedModel
        if nargin<5 || isempty(saveModelFileName)
            saveModelFileName = sprintf('trainedModel_classif_%s.mat',datestr(now,'yyyymmdd_HHMM'));
        end
        fprintf('Saving trained struct to "%s"\n', saveModelFileName);
        save(saveModelFileName, 'trainedStruct');  % everything
    end
end
