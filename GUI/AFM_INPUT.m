%% Script to Process HDF5 Files from Asylum's .ARDF Converter
% This script processes the h5 file produced by using Asylum's .ARDF converter.
% It reads the data, extracts force curves, calculates contact points, and estimates
% elastic modulus using different methods.
%
% Authors: J.H. , R. J. W., E. U. A.

%% Clear workspace, command window, and close all figures
clear;
clc;
close all;

%% Runtime Parameters: File I/O, Graphics, etc.
% Define the location of the HDF5 file to be processed
h5_file_loc ="D:\ARDF_for_Jon\231103\WTp5uM_09.h5";
% h5_file_loc = "D:\Microsopy\AFM\Patterns_Smiti_April_2021\cell_00200.h5"; % Alternate file location

% Plotting and saving options
PLOT_OPT = 0;  % 1 enables plot generation, 0 disables it
SAVE_OPT = 1;  % 1 enables saving results, 0 disables it
FontSize = 10; % Font size for plots

% Generate the save file name for the processed data
splitStr = split(h5_file_loc, '\');
fileName = splitStr(end);
SAVE_NAME = "D:\ARDF_for_Jon\231103\" + strrep(fileName, 'h5', 'mat');

% Notes on the saved results:
% F_Matrix (cell array): Force of deflection of the cantilever (for Force vs Depth)
% D_Matrix (cell array): Indentation depth vectors for each indentation
% E_Matrix (double array): Last value (deepest) of the pointwise modulus 
% Ext_Matrix (cell array): Raw extension values for each indentation
% ExtDefl_Matrix (cell array): Deflection values for each indentation
% CP_Matrix (double array): Contact points (index or extension value)
% Height_Matrix: Visualization of relative heights based on contact points
% PWE_Matrix (cell array): Pointwise modulus vectors for each indentation

%% Parameters for Finding the Contact Point
CONTACT_METHOD_OPT = 3;  % Method for contact point detection
% Options for CONTACT_METHOD_OPT:
% 1: Least square fit to a linear-quadratic piecewise function
% 2: Ratio of variance method

% Parameters for Linear-Quadratic Fit Method
NUM_PTS_CONSIDERED = 500;     % Max number of points for fitting from the end of extension
MAX_DEFL_FIT = 7.5;           % Max deflection considered for fitting (nm)
NUM_PTS_TO_AVERAGE = 500;     % Number of initial points used to estimate baseline deflection
MAX_STD_RAISE_ERROR = 1;      % Max standard deviation for baseline deflection estimation (nm)

% Parameters for Ratio of Variance Method
ROV_INTERVAL_N = 10;  % Number of samples used in the interval for the ratio of variance calculation

%% AFM Tip Parameters - No need to input spring constant; it is read from the experimental data
% https://www.nanoandmore.com/AFM-Probe-hq-xsc11-hard-al-bs - pattern
% https://www.nanoandmore.com/AFM-Probe-PNP-TR
%  normal = not high aspect, ie, not the one for smiti's micropatterns
% Tip properties based on specific models:
%R = 10;              % Radius of curvature for standard silicon nitride tips (nm)
% R = 20;            % Alternate tip radius for specific patterns
%R = 42;            % Tip radius for older Asylum AFM probes
R = 30;              % Rob's MFS experiments
%th = 35 * pi / 180;  % Cone semi-angle for normal silicon nitride probe (radians) (old and new)
% th = 20 * pi / 180; % Alternate semi-angle for specific patterns
th = 39 * pi / 180; % Rob's MFS
b = R * cos(th);     % Cylindrical radius for modulus calculation
v = 0.5;             % Poisson's ratio for the material

%% Parameters for Modulus Calculation
MODEL_QUADRATIC_FIT = 0;  % Method for modulus calculation from Force-Depth curves
% Options for MODEL_QUADRATIC_FIT:
% 0: Pointwise calculation using raw data
% 1: Pointwise calculation using quadratic fit
% 2: Compare pointwise and quadratic fit plots
% 3: Hertz contact model (single modulus value, no fit)

% Execute the main AFM analysis script
AFM_POST_JONv6;