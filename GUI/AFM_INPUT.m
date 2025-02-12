%% Script to Process HDF5 Files from Asylum's .ARDF Converter
% This script processes the h5 file produced by using Asylum's .ARDF converter.
% It reads the data, extracts force curves, calculates contact points, and estimates
% elastic modulus using different methods.
%
% Authors: J.H.,  E. U. A.

%% Clear workspace, command window, and close all figures
clear;
clc;
close all;

%% Runtime Parameters: File I/O, Graphics, etc.
% Define the location of the HDF5 file to be processed
h5_file_loc = 'D:\AFM\240731-SBL-CNI2\CSA_22.h5';
% h5_file_loc = "D:\Microsopy\AFM\Patterns_Smiti_April_2021\cell_00200.h5"; % Alternate file location

% Plotting and saving options
PLOT_OPT = 0;  % 1 enables plot generation, 0 disables it
SAVE_OPT = 1;  % 1 enables saving results, 0 disables it
FontSize = 10; % Font size for plots

% Generate the save file name for the processed data
splitStr = split(h5_file_loc, '\');
fileName = splitStr(end);
SAVE_NAME = "C:\Users\MrBes\Documents\MATLAB\Jon_AFM_Code\version4\CNI_unprocessed\240731-SBL-CNI2_" + strrep(fileName, 'h5', 'mat');

% Notes on the saved results:
% F_Matrix (cell array): Force of deflection of the cantilever (for Force vs Depth)
% D_Matrix (cell array): Indentation depth vectors for each indentation
% E_Matrix (double array): Last value (deepest) of the pointwise modulus 
% Ext_Matrix (cell array): Raw extension values for each indentation
% ExtDefl_Matrix (cell array): Deflection values for each indentation
% CP_Matrix (double array): Contact points (index or extension value)
% Height_Matrix: Visualization of relative heights based on contact points
% PWE_Matrix (cell array): Pointwise modulus vectors for each indentation

%% Set how contact point is determined
% 1: Bidomain linear-quadratic fit. Edit the hyperparameters in
% AFM_POST_Jonv6
% 2: SNAP (https://www.nature.com/articles/s41598-017-05383-0; see the SI
% for method). Edit hyperparameters in AFM_POST_Jonv6
% 3: Neural network. Specify the .mat file for the model if choosing this.

CONTACT_METHOD_OPT = 3;
networkModel = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\trainedRegressionModels\two_conv_LSTM_sequence_pooling_relu.mat";



%% AFM Tip Parameters - No need to input spring constant; it is read from the experimental data
% https://www.nanoandmore.com/AFM-Probe-hq-xsc11-hard-al-bs - pattern
% https://www.nanoandmore.com/AFM-Probe-PNP-TR
%  normal = not high aspect, ie, not the one for smiti's micropatterns
% Tip properties based on specific models:
R = 10;              % Radius of curvature for standard silicon nitride tips (nm)
% R = 20;            % Alternate tip radius for specific patterns
%R = 42;            % Tip radius for older Asylum AFM probes
%R = 30;              % Rob's MFS experiments
th = 35 * pi / 180;  % Cone semi-angle for normal silicon nitride probe (radians) (old and new)
% th = 20 * pi / 180; % Alternate semi-angle for specific patterns
%th = 39 * pi / 180; % Rob's MFS
b = R * cos(th);     % Cylindrical radius for modulus calculation
v = 0.5;             % Poisson's ratio for the material




% Execute the main AFM analysis script
AFM_POST_JONv6;