% PlotMaeScores.m
% This script loads 7 MAT files (each containing a vector named maeScoresNm),
% then produces two plots:
%   1) A swarm chart with mean ± std error bars (mean shown as open circle)
%   2) A box plot

clear;
clc;
close all;

%% Parameters
lineWidth    = 1.5;
fontName     = 'Arial';
fontSize     = 20;
markerSize   = 100;   % Marker size for individual points
jitterAmount = 0.1;   % Horizontal jitter in swarm plot

%% File paths & group names
filePaths = { ...
    'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\evaluation\regression\CNNs\two_conv_biLSTM_sequence_pooling_relu_no_augmentation_EverythingOver40nmTrimmedTo15nm.mat',...
    'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\evaluation\regression\CNNs\two_conv_biLSTM_sequence_pooling_relu_no_augmentation_EverythingOver40nmTrimmedTo20nm.mat',...
    'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\evaluation\regression\CNNs\two_conv_biLSTM_sequence_pooling_relu_no_augmentation_EverythingOver40nmTrimmedTo25nm.mat',...
    'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\evaluation\regression\CNNs\two_conv_biLSTM_sequence_pooling_relu_no_augmentation_EverythingOver40nmTrimmedTo30nm.mat',...
    'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\evaluation\regression\CNNs\two_conv_biLSTM_sequence_pooling_relu_no_augmentation_EverythingOver40nmTrimmedTo35nm.mat',...
    'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\evaluation\regression\CNNs\two_conv_biLSTM_sequence_pooling_relu_no_augmentation_EverythingOver40nmTrimmedTo40nm.mat',...
    'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\evaluation\regression\CNNs\two_conv_biLSTM_sequence_pooling_relu_no_augmentation_EverythingOver40nmNoTrimming.mat'...
};

groupNames = { '15','20','25','30','35','40','Full Depth' };

%% Load data and combine
allMaeScores   = [];
allGroupLabels = {};

for i = 1:length(filePaths)
    dataStruct    = load(filePaths{i});        % expects variable "maeScoresNm"
    maeScores     = dataStruct.maeScoresNm;  
    thisGroupName = groupNames{i};
    
    allMaeScores   = [allMaeScores; maeScores(:)];
    allGroupLabels = [allGroupLabels; repmat({thisGroupName}, numel(maeScores), 1)];
end

groups = categorical(allGroupLabels, groupNames);

%% First Figure: Swarm‐style plot with mean ± std (mean as open circle)
figure('Name','MAE Swarm + Error Bars','Color','white');
hold on;

uniqueGroups = unique(groups);
for i = 1:numel(uniqueGroups)
    grp = uniqueGroups(i);
    idx = groups == grp;
    
    % Jittered points
    xCenter = i;
    xVals   = xCenter + (rand(sum(idx),1) - 0.5)*2*jitterAmount;
    yVals   = allMaeScores(idx);
    scatter(xVals, yVals, markerSize, 'filled','Marker','square', 'MarkerFaceColor',[.4 .4 .4]);
    
    % Compute mean & std
    m = mean(yVals);
    s = std(yVals);
    
    % Plot mean as black open circle with vertical error bar
    errorbar(xCenter, m, s, 'ko', ...
             'LineWidth', lineWidth, ...
             'CapSize', 10, 'MarkerFaceColor','k');
end

% Axes formatting
xlim([0.5, numel(uniqueGroups)+0.5]);
set(gca, ...
    'XTick', 1:numel(uniqueGroups), ...
    'XTickLabel', cellstr(uniqueGroups), ...
    'FontName', fontName, ...
    'FontSize', fontSize);
xlabel('Maximum deflection (nm)');
ylabel('MAE (nm)');
grid on; box on;
ylim([20 105]);
hold off;

%% Second Figure: Boxplot
figure('Name','MAE Boxplot','Color','white');
boxplot(allMaeScores, groups, 'Notch', 'off');
xlabel('Trim / Deflection Group');
ylabel('MAE (nm)');
title('MAE per Group (Boxplot)');
set(gca, 'FontName', fontName, 'FontSize', fontSize);
grid on; box on;
