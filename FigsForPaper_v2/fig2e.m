%% SCRIPT: plot_2x2_classification_examples.m
%
% This script loads your classification data and model, identifies curves
% that are both annotated as good (1) and predicted as good, or annotated
% as bad (0) and predicted as bad, and plots 2 examples of each in a 2x2
% tiled layout.
%
% Author: ...
% Date:   ...

close all
clear
clc

%% USER-PARAMETERS
% Probability >= threshold => predicted "bad" (0/reject)
% Probability <  threshold => predicted "good" (1/accept)
threshold = 0.5;

% If these are set to real indices, we use them directly.
% If they are NaN, we pick them randomly from the correct-good/correct-bad sets.
goodNdx1 = 2018;
goodNdx2 = 3383;
badNdx1  = 305;
badNdx2  = 5774;



% The X-ticks for each subplot
%xTickValues = 0:1000:5000;

%% 1) Load the classification data
preprocessedDataFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\training\classification_processed_files\processed_features_for_classification_All.mat";
load(preprocessedDataFile, 'rawExt', 'rawDefl', 'goodOrBad');

numCurves = length(goodOrBad);
if length(rawExt) ~= numCurves || length(rawDefl) ~= numCurves
    error('Mismatch in lengths: rawExt, rawDefl, goodOrBad must match.');
end

%% 2) Load the classification model
modelFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\evaluation\classification\two_CNN_LSTMsequence_GAP_ReLu_classification.mat";
Smodel = load(modelFile);
if ~isfield(Smodel, 'results')
    error('Model file %s does not contain "results".', modelFile);
end
foldStruct = Smodel.results;  % 1x5 struct array

%% 3) Build an array of predicted probabilities for each curve
%    Then convert to predicted label (good=1, bad=0).
predProb = nan(1, numCurves);

for f = 1:numel(foldStruct)
    trInds  = foldStruct(f).trainIndices;
    teInds  = foldStruct(f).testIndices;
    trPreds = foldStruct(f).YTrainPred;  % probabilities of 'bad'
    tePreds = foldStruct(f).YTestPred;   % probabilities of 'bad'

    if length(trInds) ~= length(trPreds)
        error('Fold %d: trainIndices vs. YTrainPred length mismatch.', f);
    end
    if length(teInds) ~= length(tePreds)
        error('Fold %d: testIndices vs. YTestPred length mismatch.', f);
    end

    predProb(trInds) = trPreds;
    predProb(teInds) = tePreds;
end

predLabel = nan(1, numCurves);
predLabel(predProb >= threshold) = 0;  % bad
predLabel(predProb <  threshold) = 1;  % good

%% 4) Find the curves that are "correct" => annotated == predicted
% We'll focus on:
%    goodOrBad=1 & predLabel=1 => "accept"
%    goodOrBad=0 & predLabel=0 => "reject"
correctGoodInds = find(goodOrBad == 1 & predLabel == 1);
correctBadInds  = find(goodOrBad == 0 & predLabel == 0);

if isempty(correctGoodInds)
    error('No correct-good indices found!');
end
if isempty(correctBadInds)
    error('No correct-bad indices found!');
end

rng('shuffle');  % for random selection (if needed)

% If user didn't provide a specific index, pick one from the set
if isnan(goodNdx1)
    goodNdx1 = randsample(correctGoodInds, 1)
end
if isnan(goodNdx2)
    goodNdx2 = randsample(correctGoodInds, 1)
end
if isnan(badNdx1)
    badNdx1 = randsample(correctBadInds, 1)
end
if isnan(badNdx2)
    badNdx2 = randsample(correctBadInds, 1)
end

%% 5) Set up a 2x2 tiled layout
fig = figure('Name','2x2 Classification Examples','Color','w');
t = tiledlayout(2, 2, 'TileSpacing','compact', 'Padding','compact');

% Top row: good examples
nexttile(t,1);
plotOneCurveClassification(goodNdx1, rawExt, rawDefl, predLabel, 'k');

nexttile(t,2);
plotOneCurveClassification(goodNdx2, rawExt, rawDefl, predLabel, 'k');

% Bottom row: bad examples
nexttile(t,3);
plotOneCurveClassification(badNdx1, rawExt, rawDefl, predLabel, 'r');

nexttile(t,4);
plotOneCurveClassification(badNdx2, rawExt, rawDefl, predLabel, 'r');

% Shared X and Y labels
xlabel(t,'Extension (nm)');
ylabel(t,'Deflection (nm)');

%% 6) (Optional) Add an overall title or annotation
% title(t, sprintf('Threshold=%.2f', threshold), 'FontWeight','bold');

%% HELPER FUNCTION
function plotOneCurveClassification(curveIdx, rawExt, rawDefl, pLabel, color)
    % Retrieve raw data
    ext = rawExt{curveIdx};
    dfl = rawDefl{curveIdx};

    % Offset so extension starts at 0, deflection starts at 0
    ext0 = ext(1);
    ext = ext - ext0;
    dfl0 = min(dfl);
    dfl = dfl - dfl0;

    % Plot
    plot(ext, dfl, [color '*'], 'LineWidth', 1,'MarkerSize',2);
    hold on; grid on;

    % Title: "Prediction: Accept" or "Prediction: Reject"
    if pLabel(curveIdx) == 1
        title('Prediction: Accept');
    else
        title('Prediction: Reject');
    end

    % Set same X tick labels for each plot
    %xlim([xTickVals(1), xTickVals(end)]);
    %xlim([0 5250])
    %xticks(xTickVals);

    % We'll leave Y-limits alone; or you can unify them if you want
    hold off;
end
