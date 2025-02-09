%% Figure_Script_Normalized_TiledLayout.m
%{
  This script loads one or more .mat files (each containing Ext_Matrix and
  ExtDefl_Matrix) and plots selected force curves. Extension and deflection
  are each normalized (independently) from 0 to 1 based on the min and max
  of that particular curve.

  * The user should provide:
      1) A string array of .mat file paths in matFilePaths
      2) A row vector of row indices in rowVector
      3) A row vector of column indices in colVector

  * The number of elements in matFilePaths, rowVector, and colVector must
    match. Each entry corresponds to one subplot.

  * We will use tiledlayout to arrange the subplots in two columns.
  * Only subplots in the bottom row get an x-axis label.
  * Only subplots in the left column get a y-axis label.
  * Tick labels are kept for all subplots.
  * An overall figure title is added via sgtitle: "Anomalous Deflection Curves".
%}

clear; clc; close all;

%% 1) User Inputs 
matFilePaths = [
    "D:\AFM_PKD\211102-Gusella-PKD\NotValidated\DKO_Dish01_2021Nov02";
    "D:\AFM_PKD\211102-Gusella-PKD\NotValidated\DKO_Dish01_2021Nov02.mat";
  %"C:\Users\MrBes\Documents\MATLAB\Jon_AFM_Code\version4\breastcancerafm\2022-03-09 Cancer PTN\2022-03-09 Cancer PTN\LM24_29.mat" ;
  %"C:\Users\MrBes\Documents\MATLAB\Jon_AFM_Code\version4\breastcancerafm\2022-03-09 Cancer PTN\2022-03-09 Cancer PTN\LM24_29.mat";
  "C:\Users\MrBes\Documents\MATLAB\Jon_AFM_Code\version4\Training\MCF10a\MCF10a_25.mat";
  "C:\Users\MrBes\Documents\MATLAB\Jon_AFM_Code\version4\Training_CNI_and_PKD_Expanded\240731-SBL-CNI2_CTR_28.mat";
  "C:\Users\MrBes\Documents\MATLAB\Jon_AFM_Code\version4\Training\MCF10a\MCF10a_05.mat";
  "C:\Users\MrBes\Documents\MATLAB\Jon_AFM_Code\version4\Training\LM24\LM24_01.mat";
];
rowVector = [17; 17; 4; 1; 3; 1];
colVector = [14; 31; 3; 1; 3; 1];

    

%% 2) Basic checks
if length(matFilePaths) ~= length(rowVector) || length(matFilePaths) ~= length(colVector)
    error('Lengths of matFilePaths, rowVector, and colVector must be the same.');
end

% Number of plots is dictated by the length of these vectors
nPlots = length(matFilePaths);
nCols = 6;
nRows = ceil(nPlots / nCols);

%% 3) Create tiled layout

fontSize = 16;
figure('Name','AFM Curves (Normalized)','NumberTitle','off','Color','w');
t = tiledlayout(nRows, nCols, 'TileSpacing','compact', 'Padding','tight');
sgtitle(t, 'Anomalous Force Curves', 'FontWeight','bold','FontSize',fontSize);
t.XLabel.String = 'Normalized Extension';
t.XLabel.FontSize = fontSize;

t.YLabel.String = 'Normalized Deflection';
t.YLabel.FontSize = fontSize;


%% 4) Loop over each curve to plot
for iPlot = 1:nPlots
    
    % Load the .mat file
    load(matFilePaths(iPlot), 'Ext_Matrix', 'ExtDefl_Matrix');
    
    % Sanity check: confirm variables loaded
    if ~exist('Ext_Matrix','var') || ~exist('ExtDefl_Matrix','var')
        warning('Required variables not found in %s. Skipping...', matFilePaths(iPlot));
        continue;
    end
    
    % Extract row/col for this subplot
    thisRow = rowVector(iPlot);
    thisCol = colVector(iPlot);
    
    % Extract extension and deflection
    tmpExt  = Ext_Matrix{thisRow, thisCol};
    tmpDefl = ExtDefl_Matrix{thisRow, thisCol};
    
    % Quick checks
    if ~isvector(tmpExt) || ~isvector(tmpDefl)
        warning('Cell (%d, %d) in %s is not a vector. Skipping...', ...
            thisRow, thisCol, matFilePaths(iPlot));
        continue;
    end
    if length(tmpExt) ~= length(tmpDefl)
        warning('Cell (%d, %d) in %s has mismatched lengths. Skipping...', ...
            thisRow, thisCol, matFilePaths(iPlot));
        continue;
    end
    
    % Normalize extension and deflection from 0 to 1
    extMin = min(tmpExt);
    extMax = max(tmpExt);
    deflMin = min(tmpDefl);
    deflMax = max(tmpDefl);
    
    if (extMax - extMin) == 0 || (deflMax - deflMin) == 0
        warning('Flat data in %s at cell (%d, %d). Skipping normalization...', ...
            matFilePaths(iPlot), thisRow, thisCol);
        continue;
    end
    
    tmpExtNorm  = (tmpExt  - extMin ) / (extMax  - extMin);
    tmpDeflNorm = (tmpDefl - deflMin) / (deflMax - deflMin);

    % Select the next tile
    nexttile(iPlot);
    
    % Plot
    plot(tmpExtNorm, tmpDeflNorm, '*', ...
        'LineWidth', 1, 'MarkerSize', 2, 'MarkerEdgeColor', 'k');
    hold on; grid on;
    axis equal
    
    % Set axis to [0, 1]
    xlim([0 1]);
    ylim([0 1]);
    xticks([0:.2:1])
    yticks([0:.2:1])

    % Identify which row/column we are in for labeling
    % row index (in tiled layout) = ceil(iPlot / nCols)
    % col index (in tiled layout) = mod(iPlot-1, nCols) + 1
    currentRow = ceil(iPlot / nCols);
    currentCol = mod(iPlot-1, nCols) + 1;
    
    % Only label x-axis for the bottom row
    if currentRow == nRows
        %xlabel('Normalized Extension');
    end
    
    % Only label y-axis for the left column
    if currentCol == 1
       % ylabel('Normalized Deflection');
    end
    
    % Otherwise, we do *not* remove tick labels or anything:
    % we simply won't add an X or Y label if it's not bottom/left.
end

% Optionally adjust figure size
% set(gcf, 'Position', [100, 100, 900, 700]);
