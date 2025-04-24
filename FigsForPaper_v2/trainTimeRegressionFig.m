%% compareTrainingTimes.m
% This script compares training times from MATLAB and Python by creating a 
% publication‐worthy swarm chart.
clear;
clc;
close all;

%% Parameters
lineWidth = 1.5;
fontName = 'Arial';
fontSize = 20;

% File paths
matlabFile = 'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regressionTrainingTimes.mat';
pythonFile = 'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\SHAP\Python\execution_times.xlsx';

%% Load MATLAB training times
dataMat = load(matlabFile);  % Contains variable "trainingTimes" (10x1 double)
trainingTimesMatlab = dataMat.trainingTimes;

%% Load Python training times
% Assume the first column of the first sheet contains the training times.
T = readtable(pythonFile);
trainingTimesPython = T{:,1};

%% Combine data for plotting
% Create a grouping variable: one group for MATLAB, one for Python.
groupLabels = [repmat({'MATLAB'}, length(trainingTimesMatlab), 1); repmat({'Python'}, length(trainingTimesPython), 1)];
allTimes = [trainingTimesMatlab; trainingTimesPython];

% Create a categorical variable for groups.
groups = categorical(groupLabels, {'MATLAB', 'Python'});

%% Create figure
figure;
hold on;

% Define parameters for the swarm plot.
jitterAmount = 0.1;   % Horizontal jitter (adjust if needed)
markerSize = 100;     % Marker size for individual points

% Get unique groups and their numeric positions.
uniqueGroups = unique(groups);
for i = 1:length(uniqueGroups)
    groupName = uniqueGroups(i);
    % Logical index for this group
    idx = groups == groupName;
    % All x-values for this group are centered at its numeric value with jitter.
    xCenter = double(groupName);  % MATLAB: 1, Python: 2
    xVals = xCenter + (rand(sum(idx),1) - 0.5) * 2 * jitterAmount;
    yVals = allTimes(idx);
    
    % Plot individual training times as blue dots.
    scatter(xVals, yVals, markerSize, 'b', 'filled', 'MarkerFaceAlpha', 0.7);
    
    % Compute group mean and standard deviation.
    groupMean = mean(yVals);
    groupStd = std(yVals);
    
    % Overlay the group mean as a thick black horizontal line.
    plot([xCenter-0.15, xCenter+0.15], [groupMean, groupMean], 'k-', 'LineWidth', lineWidth);
    
    % Overlay red vertical error bars for the standard deviation.
    errorbar(xCenter, groupMean, groupStd, 'vertical', 'Color', 'r', 'LineWidth', lineWidth, 'CapSize', 10);
end

% Set x-axis limits and labels.
xlim([0.5, 2.5]);
set(gca, 'XTick', [1, 2], 'XTickLabel', {'MATLAB', 'Python'});

% Set axes properties.
set(gca, 'FontName', fontName, 'FontSize', fontSize);
ylabel('Training Time (s)');
title('Comparison of Training Times: MATLAB vs. Python');

grid on;
box on;
hold off;
