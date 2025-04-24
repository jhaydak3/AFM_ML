%% plotFineTuningResults.m
% This script loads the fine-tuning evaluation results and plots a
% swarm plot of hold‐out MAE (in nm) versus the number of training curves used
% for fine-tuning. Each group represents a different training size (e.g., 0, 25, ... 250)
% and displays the individual MAE values with an error bar showing
% the mean ± standard deviation.
%
% Note: Update the file path if needed.

clear; clc; close all;

%% ==================== Load the Results ==================== %%
resultsFile = 'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\FineTuning\spherical_test.mat';
load(resultsFile, 'results', 'curveTrainSizes');

%% ==================== Plot Parameters ==================== %%
fontName   = 'Arial';
fontSize   = 16;
markerSize = 100;
jitterAmount = 2;  % Horizontal jitter applied in the same units as training size

%% ==================== Create the Swarm Plot ==================== %%
figure('Name', 'Hold-out MAE (nm) vs Number of Training Curves', 'NumberTitle', 'off');
hold on;

% Loop over each training size defined in the curveTrainSizes vector
for idx = 1:length(curveTrainSizes)
    currentSize = curveTrainSizes(idx);
    % Find indices for this training size in the results structure
    groupIdx = find([results.curveTrainSize] == currentSize);
    if isempty(groupIdx)
        continue; % Skip if no results for this group
    end
    
    % Extract hold-out MAE values in nm for this group
    maeVals = [results(groupIdx).holdoutMAE_nm];
    
    % Create x-values for individual points with random horizontal jitter
    x_vals = currentSize + (rand(size(maeVals)) - 0.5) * 2 * jitterAmount;
    scatter(x_vals, maeVals, markerSize, 'filled', 'MarkerFaceColor', [.4 .4 .4],'Marker','s');
    
    % Calculate mean and standard deviation for error bars
    meanMAE = mean(maeVals);
    stdMAE  = std(maeVals);
    errorbar(currentSize, meanMAE, stdMAE, 'ko', 'LineWidth', 1.5, 'CapSize', 10, 'MarkerFaceColor','k');
end

%% ==================== Customize the Plot ==================== %%
xlabel('Number of curves used for fine-tuning', 'FontName', fontName, 'FontSize', fontSize);
ylabel('Contact Point MAE (nm)', 'FontName', fontName, 'FontSize', fontSize);
set(gca, 'FontName', fontName, 'FontSize', fontSize);
grid on;
box on;

hold off;

xlim([-10 275])
ylim([150 820])