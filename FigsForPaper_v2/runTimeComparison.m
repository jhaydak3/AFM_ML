%% CompareExecutionAndTrainingTimes_All.m
% This script creates swarm plots comparing MATLAB and Python timings for:
%   1. Regression Prediction Execution Time
%   2. Classification Prediction Execution Time
%   3. Preprocessing Time
%   4. Classification Training Time
%   5. Regression Training Time
%
% All figure/plot styling parameters are defined once below and then reused.

 clear; clc; close all;

%% ===================== GLOBAL PLOT PARAMETERS ========================= %%
figureWidth  = 5;          % inches
figureHeight = 5;          % inches
lineWidthMain= 1.5;        % line width for error bars, axes, etc.
fontName     = 'Arial';    % font family
fontSize     = 16;         % font size
markerSize   = 50;         % size for scatter points
jitterAmount = 0.1;        % horizontal jitter for scatter
groupLabels  = {'MATLAB','Python'};

% Colours
matColor = [0 0 1];        % Blue for MATLAB
pyColor  = [1 0.5 0];      % Orange for Python

% Helper for consistent figure creation
createFig = @(name) figure( ...
    'Name'     , name, ...
    'Units'    , 'inches', ...
    'Position' , [1 1 figureWidth figureHeight], ...
    'Color'    , 'w');

% Apply default graphics properties so every axis uses the same style
set(0,'DefaultAxesFontName' ,fontName);
set(0,'DefaultAxesFontSize' ,fontSize);
set(0,'DefaultAxesLineWidth',lineWidthMain);
set(0,'DefaultLineLineWidth',lineWidthMain);

%% ===================== 1. Regression Prediction Execution Time =========
reg_py_file  = 'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\SHAP\Python\execution_times.xlsx';
reg_mat_file = 'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\evaluationTimeRegression2.mat';

py_reg_times  = readmatrix(reg_py_file , 'Sheet', 2);
dataReg       = load(reg_mat_file);
mat_reg_times = dataReg.predictionTimes;

createFig('Regression Prediction Execution Time'); hold on;
% MATLAB group -------------
nMat = numel(mat_reg_times);
x_mat = 1 + (rand(nMat,1)-0.5)*2*jitterAmount;
scatter(x_mat, mat_reg_times, markerSize, 'filled', 'MarkerFaceColor', [ .4 .4 .4], 'Marker', 'square');

meanMat = mean(mat_reg_times);  stdMat = std(mat_reg_times);
errorbar(1, meanMat, stdMat, 'ko', 'LineWidth', lineWidthMain, 'CapSize', 10, 'MarkerFaceColor','k');

% Python group -------------
nPy  = numel(py_reg_times);
x_py = 2 + (rand(nPy ,1)-0.5)*2*jitterAmount;
scatter(x_py, py_reg_times, markerSize, 'filled', 'MarkerFaceColor', [.6 .6 .6], 'Marker', '^');

meanPy = mean(py_reg_times);  stdPy  = std(py_reg_times);
errorbar(2, meanPy, stdPy, 'ko', 'LineWidth', lineWidthMain, 'CapSize', 10, 'MarkerFaceColor','k');

xlim([0.5 2.5]);
ylim([.375 1.3])
set(gca,'XTick',[1 2],'XTickLabel',groupLabels);
ylabel('Execution Time (s)');
title('Regression Prediction Execution Time (per 1000 curves)');
grid on;  box on; hold off;

%% ===================== 2. Classification Prediction Execution Time =====
class_py_file = 'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\SHAP\Python\execution_times_classification.xlsx';
class_mat_file= 'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\classificationPredictionTimes2.mat';

py_class_times  = readmatrix(class_py_file , 'Sheet', 2);
dataClass       = load(class_mat_file);
mat_class_times = dataClass.predictionTimes;

createFig('Classification Prediction Execution Time'); hold on;
% MATLAB group -------------
nMat = numel(mat_class_times);
x_mat = 1 + (rand(nMat,1)-0.5)*2*jitterAmount;
scatter(x_mat, mat_class_times, markerSize, 'filled', 'MarkerFaceColor', [ .4 .4 .4], 'Marker', 'square');
errorbar(1, mean(mat_class_times), std(mat_class_times), 'ko', ...
         'LineWidth', lineWidthMain, 'CapSize', 10, 'MarkerFaceColor','k');

% Python group -------------
nPy  = numel(py_class_times);
x_py = 2 + (rand(nPy ,1)-0.5)*2*jitterAmount;
scatter(x_py, py_class_times, markerSize,'filled', 'MarkerFaceColor', [.6 .6 .6], 'Marker', '^');

errorbar(2, mean(py_class_times), std(py_class_times), 'ko', ...
         'LineWidth', lineWidthMain, 'CapSize', 10, 'MarkerFaceColor','k');

xlim([0.5 2.5]);
set(gca,'XTick',[1 2],'XTickLabel',groupLabels);
ylabel('Execution Time (s)');
title('Classification Prediction Execution Time (per 1000 curves)');
grid on;  box on; hold off;

%% ===================== 3. Preprocessing Time ============================
prep_py_file  = 'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\SHAP\Python\preprocess_times.xlsx';
prep_mat_file = 'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\preprocessTimes.mat';

py_prep_times  = readmatrix(prep_py_file);
dataPrep       = load(prep_mat_file);
mat_prep_times = dataPrep.preprocessingTimes;

createFig('Preprocessing Time'); hold on;
% MATLAB group -------------
nMat = numel(mat_prep_times);
x_mat = 1 + (rand(nMat,1)-0.5)*2*jitterAmount;
scatter(x_mat, mat_prep_times, markerSize, 'filled', 'MarkerFaceColor', matColor);
errorbar(1, mean(mat_prep_times), std(mat_prep_times), 'ko', ...
         'LineWidth', lineWidthMain, 'CapSize', 10);

% Python group -------------
nPy  = numel(py_prep_times);
x_py = 2 + (rand(nPy ,1)-0.5)*2*jitterAmount;
scatter(x_py, py_prep_times, markerSize, 'filled', 'MarkerFaceColor', pyColor);
errorbar(2, mean(py_prep_times), std(py_prep_times), 'ko', ...
         'LineWidth', lineWidthMain, 'CapSize', 10);

xlim([0.5 2.5]);
set(gca,'XTick',[1 2],'XTickLabel',groupLabels);
ylabel('Preprocessing Time (s)');
title('Preprocessing Time (per 1000 curves)');
grid on; grid minor; box on; hold off;

%% ===================== 4. Classification Training Time =================
train_class_py_file  = 'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\SHAP\Python\execution_times_classification.xlsx';
train_class_mat_file = 'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\classificationTrainingTimes.mat';

py_train_class_times  = readmatrix(train_class_py_file, 'Sheet', 1);
dataTrainClass        = load(train_class_mat_file);
mat_train_class_times = dataTrainClass.trainingTimes;

createFig('Classification Training Time'); hold on;
% MATLAB group -------------
nMat = numel(mat_train_class_times);
x_mat = 1 + (rand(nMat,1)-0.5)*2*jitterAmount;
scatter(x_mat, mat_train_class_times, markerSize, 'filled', 'MarkerFaceColor', matColor);
errorbar(1, mean(mat_train_class_times), std(mat_train_class_times), 'ko', ...
         'LineWidth', lineWidthMain, 'CapSize', 10);

% Python group -------------
nPy  = numel(py_train_class_times);
x_py = 2 + (rand(nPy ,1)-0.5)*2*jitterAmount;
scatter(x_py, py_train_class_times, markerSize, 'filled', 'MarkerFaceColor', pyColor);
errorbar(2, mean(py_train_class_times), std(py_train_class_times), 'ko', ...
         'LineWidth', lineWidthMain, 'CapSize', 10);

xlim([0.5 2.5]);
set(gca,'XTick',[1 2],'XTickLabel',groupLabels);
ylabel('Training Time (s)');
title('Classification Training Time');
grid on; grid minor; box on; hold off;

%% ===================== 5. Regression Training Time =====================
reg_train_py_file  = 'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\SHAP\Python\execution_times.xlsx';
reg_train_mat_file = 'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regressionTrainingTimes.mat';

py_train_reg_times  = readmatrix(reg_train_py_file , 'Sheet', 1);
dataRegTrain        = load(reg_train_mat_file);
mat_train_reg_times = dataRegTrain.trainingTimes;

createFig('Regression Training Time'); hold on;
% MATLAB group -------------
nMat = numel(mat_train_reg_times);
x_mat = 1 + (rand(nMat,1)-0.5)*2*jitterAmount;
scatter(x_mat, mat_train_reg_times, markerSize, 'filled', 'MarkerFaceColor', matColor);
errorbar(1, mean(mat_train_reg_times), std(mat_train_reg_times), 'ko', ...
         'LineWidth', lineWidthMain, 'CapSize', 10);

% Python group -------------
nPy  = numel(py_train_reg_times);
x_py = 2 + (rand(nPy ,1)-0.5)*2*jitterAmount;
scatter(x_py, py_train_reg_times, markerSize, 'filled', 'MarkerFaceColor', pyColor);
errorbar(2, mean(py_train_reg_times), std(py_train_reg_times), 'ko', ...
         'LineWidth', lineWidthMain, 'CapSize', 10);

xlim([0.5 2.5]);
set(gca,'XTick',[1 2],'XTickLabel',groupLabels);
ylabel('Training Time (s)');
title('Regression Training Time');
grid on; grid minor; box on; hold off;
