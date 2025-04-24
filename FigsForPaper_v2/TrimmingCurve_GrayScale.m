% classification_probability_vs_extension_threshold.m
%  ├─ Left Y-axis  : P(rejected) vs. extension threshold (nm)
%  └─ Right Y-axis : deflection (nm) vs. extension (nm)
% Y-axis colors now match their corresponding curves.

clear; clc; close all;
addpath('C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\helperFunctions')

%% === USER PARAMETERS ===
networkModelPath   = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\trainedClassificationModels\pooling_after_bilstm_2conv_relu_classification.mat";
dataFilePath       = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\AFM_data\Tubules\PKD_Dish01_2021Nov02.mat";
rowIdx             = 2;        % cell row for this curve
colIdx             = 23;       % cell column for this curve
increment_nm       = 50;       % extension increment for thresholding (nm)
threshold          = 0.38;     % P(rejected) threshold
textPixelOffset    = 10;       % horizontal shift of 'Reject'/'Accept' text (px)
textPixelOffsetY   = 5;        % vertical shift of 'Reject'/'Accept' text (px)

% --- style ---
lineWidthMain      = 1.5;
lineWidthGray      = 3;
figWidthInches     = 5;
figHeightInches    = 3.75;
fontSizeAxes       = 16;
legendFontSize     = 11;

colorProb       = [0.5 0.5 0.5];   % probability curve & left Y-axis
colorThreshold  = [0 0 0];         % threshold line
colorDefl       = [0 0 0];         % deflection curve & right Y-axis
colorTruncation = [0.2 0.2 0.2];   % vertical truncation line

%% === LOAD DATA ===
data  = load(dataFilePath);
Z     = data.Ext_Matrix{rowIdx, colIdx}(:);        % extension
defl  = data.ExtDefl_Matrix{rowIdx, colIdx}(:);    % deflection
CPM   = data.CP_Matrix(rowIdx, colIdx);

[~, cpIdx] = min(abs(Z - CPM));  % contact-point index
extndx     = numel(Z);           % last index

%% === PREPARE SEGMENTS ===
extRelative = Z(cpIdx:extndx) - Z(cpIdx);     % extension relative to CP
deflSeg     = defl(cpIdx:extndx);
deflShift   = deflSeg - deflSeg(1);           % shift to start at zero
maxExt      = extRelative(end);

extThresh = increment_nm:increment_nm:maxExt;
nThresh   = numel(extThresh);
pRejected = nan(nThresh, 1);

for i = 1:nThresh
    idx = find(extRelative >= extThresh(i), 1, 'first');

    % -------- correct MATLAB conditional --------
    if isempty(idx)
        cutIdx = extndx;             % use full curve if threshold beyond max
    else
        cutIdx = cpIdx - 1 + idx;    % map local index to original vector
    end
    % --------------------------------------------

    pRejected(i) = predict_NN_GUI_classification( ...
        Z(1:cutIdx), defl(1:cutIdx), networkModelPath);
end

%% === FIGURE SETUP ===
hFig = figure('Color','w', 'Units','inches', ...
              'Position',[1 1 figWidthInches figHeightInches]);
hAx  = axes('Parent',hFig, 'Box','on', 'LineWidth',lineWidthMain, ...
            'FontSize',fontSizeAxes);

%% === LEFT AXIS: probability ===
yyaxis(hAx,'left');
hAx.YAxis(1).Color = colorProb;     % match axis to curve color

hP = plot(hAx, extThresh, pRejected, '-o', ...
          'LineWidth', lineWidthMain, ...
          'MarkerSize', 6, ...
          'Color',      colorProb);
hold(hAx,'on');

yline(hAx, threshold, '--', ...
      'Color',      colorThreshold, ...
      'LineWidth',  lineWidthMain);

% Reject / Accept labels
xNorm       = 0.05;  % 5 % from left
origUnits   = hAx.Units;
hAx.Units   = 'pixels';
axPosPx     = hAx.Position;
hAx.Units   = origUnits;
dxNorm      = textPixelOffset  / axPosPx(3);
dyNorm      = textPixelOffsetY / axPosPx(4);
yRejectNorm = (threshold + 0.05 - hAx.YLim(1)) / diff(hAx.YLim);
yAcceptNorm = (threshold - 0.05 - hAx.YLim(1)) / diff(hAx.YLim);

text(hAx, xNorm - dxNorm, yRejectNorm + dyNorm, 'Reject', ...
     'Units','normalized', 'Color',colorThreshold, ...
     'FontSize',fontSizeAxes, 'VerticalAlignment','bottom');
text(hAx, xNorm - dxNorm, yAcceptNorm + dyNorm, 'Accept', ...
     'Units','normalized', 'Color',colorThreshold, ...
     'FontSize',fontSizeAxes, 'VerticalAlignment','top');

% Truncation line at last under-threshold point
lastUnder = find(pRejected < threshold, 1, 'last');
if ~isempty(lastUnder)
    xPos = extThresh(lastUnder);
    xline(hAx, xPos, 'Color',colorTruncation, 'LineWidth',lineWidthGray);
end

ylabel(hAx,'Probability curve rejected');
ylim(hAx,[0 1]);

%% === RIGHT AXIS: deflection ===
yyaxis(hAx,'right');
hAx.YAxis(2).Color = colorDefl;     % match axis to curve color

plot(hAx, extRelative, deflShift, '.', ...
     'MarkerSize', 8, ...
     'Color',      colorDefl);

ylabel(hAx,'Deflection (nm)');

%% === SHARED PROPERTIES ===
xlabel(hAx,'Extension (nm)');
xlim(hAx,[0 maxExt]);
grid(hAx,'on');

legend(hAx, {'P(rejected)','Deflection'}, ...
       'Location','best', 'FontSize',legendFontSize);

hold(hAx,'off');
