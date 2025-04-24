% classification_probability_vs_extension_threshold.m
% Left axis: P(rejected) vs. extension threshold (nm)
% Right axis: deflection (nm) vs. extension (nm)

clear; clc; close all;

%% === USER PARAMETERS ===
networkModelPath   = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\trainedClassificationModels\pooling_after_bilstm_2conv_relu_classification.mat";
dataFilePath       = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\AFM_data\Tubules\PKD_Dish01_2021Nov02.mat";
rowIdx             = 2;      % cell row for this curve
colIdx             = 23;     % cell column for this curve
increment_nm       = 50;     % extension increment for thresholding (nm)
threshold          = 0.38;   % P(rejected) threshold
textPixelOffset    = 10;     % horizontal shift of 'Reject'/'Accept' text (px)
textPixelOffsetY   = 5;      % vertical shift of 'Reject'/'Accept' text (px)
lineWidthMain      = 1.5;    % line width for main plots & axes
lineWidthGray      = 3;      % line width for the gray truncation line
figWidthInches     = 5;      % figure width (inches)
figHeightInches    = 3.75;   % figure height (inches)
fontSizeAxes       = 16;     % font size for axes labels, ticks, and text
legendFontSize     = 11;     % font size for legend
colorProb          = [0 0.4470 0.7410];   % color for P(rejected) curve
colorThreshold     = [1 0 0];             % color for threshold line
colorDefl          = [0 0 0];             % color for deflection curve (black)
colorTruncation    = [0.5 0.5 0.5];       % color for truncation vertical line

%% === LOAD DATA ===
data = load(dataFilePath);
ExtM = data.Ext_Matrix;
DeflM = data.ExtDefl_Matrix;
CPM = data.CP_Matrix;

Z = ExtM{rowIdx,colIdx}(:);
defl = DeflM{rowIdx,colIdx}(:);
[~, cpIdx] = min(abs(Z - CPM(rowIdx,colIdx)));
extndx = numel(Z);

%% === PREPARE SEGMENTS ===
extRelative = Z(cpIdx:extndx) - Z(cpIdx);
deflSeg     = defl(cpIdx:extndx);
deflShift   = deflSeg - deflSeg(1);
maxExt      = extRelative(end);

extThresh = increment_nm:increment_nm:maxExt;
nThresh   = numel(extThresh);
pRejected = nan(nThresh,1);

for i = 1:nThresh
    idx = find(extRelative >= extThresh(i),1,'first');
    if isempty(idx)
        cutIdx = extndx;
    else
        cutIdx = cpIdx - 1 + idx;
    end
    pRejected(i) = predict_NN_GUI_classification(...
        Z(1:cutIdx), defl(1:cutIdx), networkModelPath);
end

%% === FIGURE SETUP ===
hFig = figure( ...
    'Color','white', ...
    'Units','inches', ...
    'Position',[1 1 figWidthInches figHeightInches] ...
);
hAx = axes( ...
    'Parent',hFig, ...
    'Box','on', ...
    'FontSize',fontSizeAxes, ...
    'LineWidth',lineWidthMain ...
);

%% === PLOT ===
yyaxis(hAx,'left');
hP = plot(hAx, extThresh, pRejected, '-o', ...
    'LineWidth',lineWidthMain, 'MarkerSize',6, 'Color',colorProb);
hold(hAx,'on');
hL = yline(hAx, threshold, '--', ...
    'Color',colorThreshold, 'LineWidth',lineWidthMain);

% Normalized positions for Reject/Accept text
xNorm = (maxExt*0.05 - hAx.XLim(1)) / diff(hAx.XLim);
yRejectNorm = (threshold + 0.05 - hAx.YLim(1)) / diff(hAx.YLim);
yAcceptNorm = (threshold - 0.05 - hAx.YLim(1)) / diff(hAx.YLim);

% Convert pixel offsets to normalized units
origUnits = hAx.Units;
hAx.Units   = 'pixels';
axPosPx     = hAx.Position;  
hAx.Units   = origUnits;
dxNorm      = textPixelOffset  / axPosPx(3);
dyNorm      = textPixelOffsetY / axPosPx(4);

text(hAx, xNorm-dxNorm, yRejectNorm+dyNorm, 'Reject', ...
    'Units','normalized', 'Color',colorThreshold, ...
    'FontSize',fontSizeAxes, 'VerticalAlignment','bottom');
text(hAx, xNorm-dxNorm, yAcceptNorm+dyNorm, 'Accept', ...
    'Units','normalized', 'Color',colorThreshold, ...
    'FontSize',fontSizeAxes, 'VerticalAlignment','top');

% Truncation line at last under-threshold point
lastUnder = find(pRejected < threshold, 1, 'last');
if ~isempty(lastUnder)
    xPos = extThresh(lastUnder);
    xline(hAx, xPos, 'Color',colorTruncation, 'LineWidth',lineWidthGray);
end
hold(hAx,'off');

ylabel(hAx, 'Probability curve rejected', 'FontSize',fontSizeAxes);
ylim(hAx, [0 1]);

yyaxis(hAx,'right');
hD = plot(hAx, extRelative, deflShift, '.', ...
    'MarkerSize',8, 'LineWidth',lineWidthMain, 'Color',colorDefl);
hAx.YAxis(2).Color = colorDefl;
ylabel(hAx, 'Deflection (nm)', 'FontSize',fontSizeAxes);

xlabel(hAx, 'Extension (nm)', 'FontSize',fontSizeAxes);
xlim(hAx, [0 maxExt]);
grid(hAx,'on');

leg = legend(hAx, [hP, hL, hD], ...
    'P(rejected)', sprintf('Threshold (%.2f)',threshold), 'Deflection', ...
    'Location','best');
leg.FontSize = legendFontSize;
