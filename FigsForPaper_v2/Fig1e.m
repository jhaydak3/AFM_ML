%% Figure_Script_Normalized_TiledLayout_with_Individual_SVGs.m
%{
  This script loads one or more .mat files (each containing Ext_Matrix and
  ExtDefl_Matrix) and plots selected force curves in a tiled layout. It also
  exports each subplot as a standalone SVG in a folder "fig1e".

  * Only the first subplot shows axis labels and tick numbers.
  * All curves are plotted with a solid line.
%}

clear; clc; close all;

%% 1) User Inputs 
matFilePaths = [
    "D:\AFM_PKD\211102-Gusella-PKD\NotValidated\DKO_Dish01_2021Nov02.mat";
    "C:\Users\MrBes\Documents\MATLAB\Jon_AFM_Code\version4\Training\MCF10a\MCF10a_05.mat";
    "C:\Users\MrBes\Documents\MATLAB\Jon_AFM_Code\version4\Training\MCF10a\MCF10a_25.mat";
    "C:\Users\MrBes\Documents\MATLAB\Jon_AFM_Code\version4\Training_CNI_and_PKD_Expanded\240731-SBL-CNI2_CTR_28.mat";
    "D:\AFM_PKD\211102-Gusella-PKD\NotValidated\DKO_Dish01_2021Nov02.mat";
    "C:\Users\MrBes\Documents\MATLAB\Jon_AFM_Code\version4\Training\LM24\LM24_01.mat";
];
rowVector = [17; 3; 4; 1; 17; 1];
colVector = [14; 3; 3; 1; 31; 1];

subplotTitles = {
    {'Tubule'}, ...
    {'MCF10a'}, ...
    {'MCF10a'}, ...
    {'Podocyte'}, ...
    {'Tubule'}, ...
    {'LM24'}
};

%% 2) Basic checks
nPlots = numel(matFilePaths);
if nPlots ~= numel(rowVector) || nPlots ~= numel(colVector) || nPlots ~= numel(subplotTitles)
    error('All input arrays must have the same length.');
end
nCols = 6;
nRows = ceil(nPlots / nCols);

%% 3) Plot style parameters
figW = 3;           % figure width (inches)
figH = 3;           % figure height (inches)
fontName = 'Arial';
fontSizeTick = 12;      % tick label size
fontSizeLabel = 16;     % axis label size
fontSizeSubTitle = 14;  % subplot title size
fontSizeOverallTitle = 20;
lineWidth = 1.5;

%% 4) Create output directory for individual SVGs
outDir = 'fig1e';
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

%% 5) Create tiled layout figure
figure('Units','inches','Position',[1 1 figW figH],'Color','w');
t = tiledlayout(nRows, nCols, 'TileSpacing','compact', 'Padding','tight');
sgtitle(t, 'Anomalous Force Curves', 'FontWeight','bold', 'FontSize', fontSizeOverallTitle);

for iPlot = 1:nPlots
    % Load data
    S = load(matFilePaths(iPlot), 'Ext_Matrix', 'ExtDefl_Matrix');
    tmpExt  = S.Ext_Matrix{rowVector(iPlot), colVector(iPlot)};
    tmpDefl = S.ExtDefl_Matrix{rowVector(iPlot), colVector(iPlot)};
    % Normalize
    tmpExtNorm  = (tmpExt  - min(tmpExt))  / (max(tmpExt)  - min(tmpExt));
    tmpDeflNorm = (tmpDefl - min(tmpDefl)) / (max(tmpDefl) - min(tmpDefl));
    
    % --- Plot in tiled layout ---
    ax = nexttile(iPlot);
    plot(ax, tmpExtNorm, tmpDeflNorm, '-', 'LineWidth', lineWidth, 'Color', 'k');
    hold(ax, 'on'); grid(ax, 'on'); axis(ax, 'equal');
    xlim(ax, [0 1]); ylim(ax, [0 1]);
    title(ax, subplotTitles{iPlot}, 'FontSize', fontSizeSubTitle, 'FontName', fontName);
    ax.FontName = fontName; ax.FontSize = fontSizeTick;
    
    if iPlot == 1
        xlabel(ax, 'Normalized Extension', 'FontSize', fontSizeLabel, 'FontName', fontName);
        ylabel(ax, 'Normalized Deflection', 'FontSize', fontSizeLabel, 'FontName', fontName);
    else
        ax.XTickLabel = [];
        ax.YTickLabel = [];
    end
    
    %% --- Export individual subplot as SVG ---
    f2 = figure('Units','inches','Position',[1 1 figW figH],'Color','w');
    ax2 = axes(f2);
    plot(ax2, tmpExtNorm, tmpDeflNorm, '-', 'LineWidth', lineWidth, 'Color', 'k');
    hold(ax2, 'on'); grid(ax2, 'on'); axis(ax2, 'equal');
    xlim(ax2, [0 1]); ylim(ax2, [0 1]);
    title(ax2, subplotTitles{iPlot}, 'FontSize', fontSizeSubTitle, 'FontName', fontName);
    ax2.FontName = fontName; ax2.FontSize = fontSizeTick;
    if iPlot == 1
        xlabel(ax2, 'Normalized Extension', 'FontSize', fontSizeLabel, 'FontName', fontName);
        ylabel(ax2, 'Normalized Deflection', 'FontSize', fontSizeLabel, 'FontName', fontName);
    else
        ax2.XTickLabel = [];
        ax2.YTickLabel = [];
    end
    saveas(f2, fullfile(outDir, sprintf('subplot%d.svg', iPlot)), 'svg');
    close(f2);
end
