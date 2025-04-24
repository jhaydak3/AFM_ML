% FineTunePlot_MergedNoDuplicates.m
% Combines two fine-tuning result files, omitting duplicate size-0 entries,
% and plots Hold-out MAE vs # curves with jitter + error bars.

clear; clc; close all;

%% ------------------------ STYLE PARAMETERS ------------------------ %%
figW         = 5;    figH  = 5;
fontName     = 'Arial';  fontSize = 16;
lw           = 1.5;  titleStr = '';
markerColor  = [.4 .4 .4];  jitterAmount = 0.2;  markerSize = 75;

%% ------------------------ LOAD RESULTS --------------------------- %%
filesToLoad = {
    'EvrenFineTuneResults2.mat'
    'EvrenFineTuneResults.mat'
};

allSizes = [];   % concatenated curveTrainSize
allMAEs  = [];   % concatenated holdoutMAE_nm

for f = 1:numel(filesToLoad)
    S = load(filesToLoad{f}, 'results');        % grab only the struct you need
    sz  = [S.results.curveTrainSize];
    mae = [S.results.holdoutMAE_nm];

    % --- Keep size-0 block only from the *first* file -------------------
    if f > 1
        keep = sz ~= 0;        % discard size-0 rows from subsequent files
        sz  = sz( keep );
        mae = mae(keep);
    end

    allSizes = [allSizes, sz];
    allMAEs  = [allMAEs,  mae];
end

allSizes = allSizes(:);     % column vectors for convenience
allMAEs  = allMAEs(:);

%% ------------------------ STATS ----------------------------------- %%
uniqueSizes = unique(allSizes, 'stable');  % keep first-seen order
meanMAE = arrayfun(@(s) mean(allMAEs(allSizes==s)), uniqueSizes);
stdMAE  = arrayfun(@(s)  std(allMAEs(allSizes==s)), uniqueSizes);

%% ------------------------ PLOT ------------------------------------ %%
figure('Units','inches','Position',[1 1 figW figH],'Color','w');
hold on; box on; grid on;

% jittered scatter
xJitter = allSizes + (rand(size(allMAEs)) - 0.5)*2*jitterAmount;
scatter(xJitter, allMAEs, markerSize, markerColor,'s', 'filled');

% error bars (mean ± std)
errorbar(uniqueSizes, meanMAE, stdMAE, ...
         'ko', 'LineWidth', lw, 'CapSize', 10);

% axis styling
ax = gca;
ax.FontName = fontName;
ax.FontSize = fontSize;
xlabel('Number of curves for fine-tuning');
ylabel('Hold-out MAE (nm)');
title(titleStr);
ylim([0 70]);

hold off;
