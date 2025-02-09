function trainedStruct = trainOneDataset(layers, dataFile, numSamplesToTrain, saveModelFile)
    % TRAINONEDATASET
    %
    % If #samples >= numSamplesToTrain => pick exactly that many => leftover => store in selfTestIndices
    %   modelForOthers = net, modelForSelf = same net, leftover used for self test
    % If #samples < numSamplesToTrain => do old logic:
    %   modelForOthers => all data
    %   modelForSelf   => 80/20
    %
    % We always skip: if leftover is empty => no self test
    %
    % Outputs a struct:
    %   .modelForOthers
    %   .modelForSelf
    %   .selfTestIndices
    %   .trainIndicesUsed
    %

    fprintf('Loading dataset "%s"...\n', dataFile);
    S = load(dataFile, 'X','Y','goodOrBad');
    if ~isfield(S,'X') || ~isfield(S,'Y')
        warning('No X or Y => returning empty struct.');
        trainedStruct = struct();
        return;
    end

    Xfull = S.X; 
    Yfull = S.Y;
    N = size(Xfull,3);
    fprintf('  # curves = %d.\n', N);

    trainedStruct = struct(...
        'modelForOthers', [], ...
        'modelForSelf',   [], ...
        'selfTestIndices', [], ...
        'trainIndicesUsed', []);

    if N<1
        warning('No curves => empty struct.');
        return;
    end

    rng(1337,'twister');

    if N >= numSamplesToTrain
        %% Large dataset
        permAll = randperm(N);
        trainInd    = permAll(1:numSamplesToTrain);
        leftoverInd = permAll(numSamplesToTrain+1:end);

        fprintf('  Using %d for training "modelForOthers", leftover=%d.\n',...
            numSamplesToTrain, length(leftoverInd));

        % ---------------------------------------------------
        % 1) Extract training set
        % ---------------------------------------------------
        Xtrain = Xfull(:,:,trainInd);
        Ytrain = Yfull(trainInd);

        % ---------------------------------------------------
        % 2) AUGMENT the training set
        %    (Pad = 100 as requested)
        % ---------------------------------------------------
        [XtrainAll, YtrainAll] = augmentData(Xtrain, Ytrain, 100);



        fprintf('  After augmentation: original=%d, augmented=%d => total=%d.\n',...
                size(Xtrain,3), size(Xaug,3), size(XtrainAll,3));

        % ---------------------------------------------------
        % 3) Train the model
        % ---------------------------------------------------
        netO = trainModelCore(layers, XtrainAll, YtrainAll);

        % Store in trainedStruct
        trainedStruct.modelForOthers = netO;
        trainedStruct.modelForSelf   = netO;  % (reuse same net for "self")
        trainedStruct.selfTestIndices= leftoverInd; 
        trainedStruct.trainIndicesUsed= trainInd;

    else
        %% Small dataset: < numSamplesToTrain
        fprintf('  < %d => use ALL for modelForOthers.\n', numSamplesToTrain);

        allInd = 1:N;
        Xtrain = Xfull;
        Ytrain = Yfull;

        % Optional: also augment even if the dataset is small
        [XtrainAll, YtrainAll] = augmentData(Xtrain, Ytrain, 100);


        netO = trainModelCore(layers, XtrainAll, YtrainAll);

        trainedStruct.modelForOthers   = netO;
        trainedStruct.trainIndicesUsed = allInd;

        % Also 80/20 for self test
        rp2 = randperm(N);
        numTrain2 = max(1, floor(0.8*N));
        trainInd2 = rp2(1:numTrain2);
        testInd2  = rp2(numTrain2+1:end);

        fprintf('  Also creating modelForSelf => train=%d, leftover=%d.\n',...
            numTrain2, length(testInd2));

        Xtrain2 = Xfull(:,:, trainInd2);
        Ytrain2 = Yfull(trainInd2);

        % Possibly augment *this* subset for "self"
        [XtrainAll2, YtrainAll2] = augmentData(Xtrain2, Ytrain2, 100);


        netS = trainModelCore(layers, XtrainAll2, YtrainAll2);

        trainedStruct.modelForSelf   = netS;
        trainedStruct.selfTestIndices= testInd2;
    end

    if ~isempty(saveModelFile)
        fprintf('Saving struct to "%s"...\n', saveModelFile);
        save(saveModelFile,'trainedStruct');
    end
end
