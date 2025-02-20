function trainedNet = trainModelCore_classification(layers, Xtrain, Ytrain)
% TRAINMODELCORE_CLASSIFICATION
% Trains a classification CNN. We:
%   1) Convert numeric labels => categorical('reject','accept') if needed.
%   2) Shuffle data
%   3) Call trainnet(...) with crossentropy
%   4) Return the dlnetwork (or [] if error)
%
% Inputs:
%   layers -> layer array
%   Xtrain -> [features x seqLen x N]
%   Ytrain -> Nx1 (0/1, or 'reject'/'accept')
%
% Output:
%   trainedNet -> dlnetwork or empty

    % Convert Y => categorical
    Ycat = toCategoricalRejectAccept(Ytrain);

    if numel(Ycat) < 2
        warning('Not enough training samples => returning empty.');
        trainedNet = [];
        return;
    end

    % Build dlnetwork
    net = dlnetwork(layers);

    % Permute X => 'CBT'
    % i.e. from [features x seqLen x N] => [features x N x seqLen]
    Xp  = permute(Xtrain,[1 3 2]);
    dlX = dlarray(Xp,'CBT');

    % Shuffle
    rp = randperm(size(dlX,2));
    dlX   = dlX(:,rp,:);
    Ycat  = Ycat(rp);

    % trainingOptions
    opts = trainingOptions('adam',...
        'MaxEpochs',30,...
        'MiniBatchSize',64*.5,...
        'Shuffle','every-epoch',...
        'Verbose',true,...
        'Plots','none',...
        'ValidationFrequency',50,...
        'InitialLearnRate',1e-4,...
        'LearnRateSchedule','piecewise',...
        'LearnRateDropFactor',1,...
        'LearnRateDropPeriod',1,...
        'Metrics',["precision","accuracy","auc","fscore","recall"],...
        'ObjectiveMetricName',"auc",...
        'OutputNetwork','last-iteration',...
        'ExecutionEnvironment','gpu',...
        'L2Regularization',1e-4);

    try
        [trainedNet, info] = trainnet(dlX, Ycat, net, "crossentropy", opts);
    catch ME
        warning('Error training => returning empty net: %s', ME.message);
        trainedNet = [];
    end
end