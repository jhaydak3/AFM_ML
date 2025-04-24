%% CombinedSwarmChartShapValues_Modified.m
% This script reads SHAP values for regression and classification from two Excel 
% files and produces separate, publication‑worthy swarm charts for each feature.
% For each feature, individual observations are plotted with jitter.
% The overlay displays the mean and standard deviation using errorbar().

clear; clc; close all;

%% Parameters

% File paths
regFilePath    = 'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\SHAP\Python\shap_values_global.xlsx';
classFilePath  = 'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\SHAP\Python\shap_values_global_classification.xlsx';

% Styling
lineWidth      = 1.5;      % width of mean/std errorbar
fontName       = 'Arial';
fontSize       = 20;       % axis labels & title
scatterColor   = [.4 .4 .4];      % point color
overlayColor   = 'k';      % mean/errorbar color
markerType = 's';
% Jitter and marker size
jitterAmount   = 0.2;     % horizontal jitter half‑width
markerSize     = 50;       % scatter marker size


% Axis labels and titles
xAxisLabel     = 'Feature';
yAxisLabel     = 'SHAP Value';
regTitleText   = '';
classTitleText = '';

% Feature names
% 
% featureNames   = { ...
%     'Deflection', ...
%     'Smoothed Deflection', ...
%     'Smoothed Derivative', ...
%     'RoV', ...
%     'Local Slope', ...
%     'Local R^2'  ...
% };
featureNames   = { ...
    'Feat1', ...
    'Feat2', ...
    'Feat3', ...
    'Feat4', ...
    'Feat5', ...
    'Feat6'  ...
};



%% Prepare display names (split two‑word names onto two lines)
featureDisplayNames = cellfun( ...
    @(s) strrep(s, ' ', '\newline'), ...
    featureNames, 'UniformOutput', false);

%% Read data
T_reg            = readtable(regFilePath);
shapValues_reg   = table2array(T_reg);
[numObs_reg, numFeatures] = size(shapValues_reg);

T_class          = readtable(classFilePath);
shapValues_class = table2array(T_class);
numObs_class     = size(shapValues_class, 1);

%% Plot Regression SHAP Swarm Chart
figure('Name', regTitleText, 'NumberTitle','off','Color','w');
hold on;
for i = 1:numFeatures
    x = i + (rand(numObs_reg,1) - 0.5)*2*jitterAmount;
    y = shapValues_reg(:,i);
    scatter(x, y, markerSize, 'filled', ...
        'MarkerFaceColor', scatterColor, 'MarkerFaceAlpha', .7, 'Marker', markerType);
    m = mean(y);
    s = std(y);
    errorbar(i, m, s, 'o', ...
        'Color', overlayColor, ...
        'MarkerFaceColor', overlayColor, ...
        'LineWidth', lineWidth, ...
        'CapSize', 10);
end

% Format axes
set(gca, ...
    'XTick',           1:numFeatures, ...
    'XTickLabel',      featureDisplayNames, ...
    'TickLabelInterpreter', 'tex', ...
    'FontName',        fontName, ...
    'FontSize',        fontSize);
xlim([0.5, numFeatures+0.5]);
xtickangle(45);

% Labels & title
xlabel(xAxisLabel, 'FontName', fontName, 'FontSize', fontSize);
ylabel(yAxisLabel, 'FontName', fontName, 'FontSize', fontSize);
title(regTitleText, 'FontName', fontName, 'FontSize', fontSize);

grid on; box on;
hold off;

%% Plot Classification SHAP Swarm Chart
figure('Name', classTitleText, 'NumberTitle','off','Color','w');
hold on;
for i = 1:numFeatures
    x = i + (rand(numObs_class,1) - 0.5)*2*jitterAmount;
    y = shapValues_class(:,i);
    scatter(x, y, markerSize, 'filled', ...
        'MarkerFaceColor', scatterColor, 'MarkerFaceAlpha', 0.7, 'Marker',markerType);
    m = mean(y);
    s = std(y);
    errorbar(i, m, s, 'o', ...
        'Color', overlayColor, ...
        'MarkerFaceColor', overlayColor, ...
        'LineWidth', lineWidth, ...
        'CapSize', 10);
end

% Format axes
set(gca, ...
    'XTick',           1:numFeatures, ...
    'XTickLabel',      featureDisplayNames, ...
    'TickLabelInterpreter', 'tex', ...
    'FontName',        fontName, ...
    'FontSize',        fontSize);
xlim([0.5, numFeatures+0.5]);
xtickangle(45);

% Labels & title
xlabel(xAxisLabel, 'FontName', fontName, 'FontSize', fontSize);
ylabel(yAxisLabel, 'FontName', fontName, 'FontSize', fontSize);
title(classTitleText, 'FontName', fontName, 'FontSize', fontSize);

grid on; box on;
hold off;
