%% CombinedSwarmAndBarChartAblationResults_Simplified.m
% This script reads:
%  1) A wide-format Excel file of regression ablation results in nm
%     (each column is a scenario, each row is one test sample's error)
%     and plots a swarm chart.
%  2) A long-format Excel file of classification ablation results with
%     two columns: "Scenario" (strings) and "AUC" (numeric).
%     It then plots a bar chart with the scenario names on the x-axis.
%
% For both plots, the x-axis labels are set to a common list:
% 'Baseline','Deflection','Smoothed Defl','Derivative','RoV','Slope','R^2'

clear; clc; close all;

%% File paths
regFilePath   = 'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\SHAP\Python\ablation_results_nm.xlsx';
classFilePath = 'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\SHAP\Python\ablation_results_auc_classification.xlsx';

%% Common styling parameters
lineWidth      = 1.5;    
thickLineWidth = 3;      
fontName       = 'Arial';
fontSize       = 20;
jitterAmount   = 0.15;   
markerSize     = 50;     
plotColor      = 'b';   % Set the plot color to blue

% Define common x-axis labels for both plots (in the desired order)
customLabels =  {'Baseline','Deflection','Smoothed Deflection','Smoothed Derivative','RoV','Local Slope','Local R^2'};

%% === 1) Regression Data (Wide Format, nm) ===
% Read the entire file as a table, preserving original column headers.
T_reg = readtable(regFilePath, 'VariableNamingRule','preserve');

% Convert to array of numeric data
ablationValues_reg = table2array(T_reg);

% Number of scenarios and observations
numScenarios_reg = size(ablationValues_reg, 2);
numObservations_reg = size(ablationValues_reg, 1);

% Create a figure for the regression swarm chart
figure('Name','Regression Ablation Results (nm)');
hold on;

for i = 1:numScenarios_reg
    % Generate jittered x-coordinates for each observation
    x = i + (rand(numObservations_reg, 1) - 0.5) * 2 * jitterAmount;
    y = ablationValues_reg(:, i);
    
    % Plot individual data points using the parameterized color
    scatter(x, y, markerSize, 'filled', 'MarkerFaceColor', plotColor, 'MarkerFaceAlpha', 0.6);
    
    % Compute group mean and standard deviation
    meanVal = mean(y);
    stdVal  = std(y);
    
    % Use errorbar to plot the mean (with a marker) and its standard deviation
    errorbar(i, meanVal, stdVal, 'o', 'Color', 'k', ...
        'LineWidth', lineWidth, 'CapSize', 10, 'MarkerFaceColor', 'k', 'MarkerSize', 8);
end

% Use common x-axis labels
set(gca, 'XTick', 1:numScenarios_reg, 'XTickLabel', customLabels);
xlim([0.5, numScenarios_reg + 0.5]);
xtickangle(45);
set(gca, 'FontName', fontName, 'FontSize', fontSize);
ylabel('Absolute Error (nm)');
title('Regression Ablation Results (nm)');
grid on; grid minor; box on;
hold off;
ylim([-200, 3500])

%% === 2) Classification Data (Long Format, AUC) ===
% Read the classification file (assumed to have two columns:
% first column "Scenario" and second column "AUC")
T_class = readtable(classFilePath, 'VariableNamingRule','preserve');

% Get the AUC values (second column)
AUC_values = T_class{:,2};

% Create a figure for the classification bar chart
figure('Name','Classification Ablation Results');
bar(AUC_values, 'FaceColor', plotColor);

% Apply common x-axis labels
set(gca, 'XTick', 1:numel(AUC_values), 'XTickLabel', customLabels);
xtickangle(45);
set(gca, 'FontName', fontName, 'FontSize', fontSize);
ylabel('AUC');
title('Classification Ablation Results');
grid on; box on; grid minor
