function [trainedNet, leftoverData] = trainClassificationOneDataset_singleModel( ...
    layers, dataFile, numSamplesToTrain)
% TRAINCLASSIFICATIONONEDATASET_SINGLEMODEL
%
% 1) Loads dataFile => X, goodOrBad
% 2) If #samples >= numSamplesToTrain => pick exactly that many for train,
%    leftover => test
% 3) If #samples < numSamplesToTrain => 80/20 split
% 4) Train a single CNN model
% 5) Return leftoverData for self-test (Xtest, Ytest)
%
% If data < 2 => no model
%

    leftoverData = struct('Xtest',[],'Ytest',[]);

    fprintf('Loading data from "%s"...\n', dataFile);
    S = load(dataFile, 'X','goodOrBad');
    if ~isfield(S,'X') || ~isfield(S,'goodOrBad')
        warning('Data file missing X/goodOrBad => returning empty.');
        trainedNet = [];
        return;
    end

    Xfull = S.X;
    Yfull = S.goodOrBad;
    N = size(Xfull,3);
    fprintf('Dataset has %d curves.\n', N);

    if N < 2
        warning('Not enough data => returning empty model.');
        trainedNet = [];
        return;
    end

    rng(1337,'twister');  % reproducible

    if N >= numSamplesToTrain
        % pick exactly numSamplesToTrain
        perm = randperm(N);
        trainInd   = perm(1:numSamplesToTrain);
        leftoverInd= perm(numSamplesToTrain+1:end);
        fprintf('Using %d for train, leftover: %d\n', ...
            numSamplesToTrain, length(leftoverInd));
    else
        % 80/20
        fprintf('Dataset < %d => 80/20 split.\n', numSamplesToTrain);
        perm = randperm(N);
        numTrain = max(1, floor(0.8*N));
        trainInd = perm(1:numTrain);
        leftoverInd = perm(numTrain+1:end);
        fprintf('Using %d for train, leftover: %d\n', ...
            numTrain, length(leftoverInd));
    end

    Xtrain = Xfull(:,:,trainInd);
    Ytrain = Yfull(trainInd);

    leftoverData.Xtest = Xfull(:,:, leftoverInd);
    leftoverData.Ytest = Yfull(leftoverInd);

    if numel(Ytrain)<2
        warning('Not enough train curves => empty model returned.');
        trainedNet = [];
        leftoverData = struct('Xtest',[],'Ytest',[]);
        return;
    end

    % Convert 0=>reject,1=>accept => categorical
    strTrain = strings(numel(Ytrain),1);
    strTrain(Ytrain==0)="reject";
    strTrain(Ytrain==1)="accept";
    YcatTrain = categorical(strTrain, {'reject','accept'});

    net = dlnetwork(layers);

    XtrainPerm = permute(Xtrain,[1,3,2]);
    dlXtrain   = dlarray(XtrainPerm,'CBT');

    rp = randperm(size(dlXtrain,2));
    dlXtrain = dlXtrain(:,rp,:);
    YcatTrain= YcatTrain(rp);

    opts = trainingOptions('adam',...
        'MaxEpochs',30,...
        'MiniBatchSize',64,...
        'Shuffle','every-epoch',...
        'Verbose',true,...
        'Plots','none',...
        'ValidationFrequency',50,...
        'InitialLearnRate',1e-3);

    fprintf('Training single CNN model...\n');
    try
        [trainedNet,info] = trainnet(dlXtrain, YcatTrain, net, "crossentropy", opts);
        disp('Training completed.');
    catch ME
        warning('Error training: %s => returning empty model.', ME.message);
        trainedNet=[];
        leftoverData=struct('Xtest',[],'Ytest',[]);
    end
end
