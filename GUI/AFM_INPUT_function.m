% run_all_h5.m
% This script searches the specified folder for all .h5 files and runs the
% function AFM_INPUT_function on each one.
addpath('C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\GUI')
% Specify the folder path
%folderPath = 'D:\testAFM';
folderPath = 'D:\Rob_Tissue_AFM\210617';
%folderPath = 'D:\AFM\240731-SBL-CNI2';
% Get a list of all .h5 files in the folder
h5Files = dir(fullfile(folderPath, '*.h5'));

% Preallocate an error log cell array (one element per file)
errorLog = cell(length(h5Files), 1);

% Process files in parallel using parfor
for k = 1:length(h5Files)
    % Build the full file path
    filePath = fullfile(h5Files(k).folder, h5Files(k).name);
    fprintf('Processing file: %s\n', filePath);
    
    % Try running the helper function; log any error messages
    try
        AFM_INPUT_function_helper(filePath);
    catch ME
        % Save the error message along with the file name
        errorLog{k} = sprintf('Error processing file %s: %s', filePath, ME.message);
        fprintf('Something wrong when processing file %s. Skipping...\n', filePath);
    end
end

% After the parfor loop, write the error log to a text file.
logFileName = 'errorLog.txt';
fid = fopen(logFileName, 'w');
if fid == -1
    warning('Could not open %s for writing.', logFileName);
else
    for k = 1:length(errorLog)
        if ~isempty(errorLog{k})
            fprintf(fid, '%s\n', errorLog{k});
        end
    end
    fclose(fid);
    fprintf('Error log written to %s\n', logFileName);
end


function AFM_INPUT_function_helper(h5_file_loc)

%% Script to Process HDF5 Files from Asylum's .ARDF Converter
% This script processes the h5 file produced by using Asylum's .ARDF converter.
% It reads the data, extracts force curves, calculates contact points, and estimates
% elastic modulus using different methods.
%
% Authors: J.H.,  E. U. A.

%% Clear workspace, command window, and close all figures


%% Runtime Parameters: File I/O, Graphics, etc.
% Define the location of the HDF5 file to be processed
%h5_file_loc ="D:\AFM\240731-SBL-CNI2\VCS_36.h5";
% h5_file_loc = "D:\Microsopy\AFM\Patterns_Smiti_April_2021\cell_00200.h5"; % Alternate file location

% Plotting and saving options
PLOT_OPT = 0;  % 1 enables plot generation, 0 disables it
SAVE_OPT = 1;  % 1 enables saving results, 0 disables it
FontSize = 10; % Font size for plots

% Generate the save file name for the processed data
splitStr = split(h5_file_loc, '\');
fileName = splitStr(end);
%SAVE_NAME = "D:\testAFM\" + strrep(fileName, 'h5', 'mat');
SAVE_NAME = "D:\Rob_Tissue_AFM\test\" + strrep(fileName, 'h5', 'mat');
%SAVE_NAME = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\GUI\AFM_analysis\CNI_predicted_with_NN\240731-SBL-CNI2_" + strrep(fileName, 'h5', 'mat');
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
networkModel = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\trainedRegressionModels\pooling_after_bilstm_2conv_relu.mat";

%% Parameters for Hertzian modulus
HERTZIAN_FRONT_REMOVE = 100; % This is the amount of depth that isn't included for Hertzian fitting.

%% Curve quality prediction
% Do you want to use a neural network to predict if the curve is good or
% bad quality?

PREDICT_QUALITY_OPT = 1;
networkModelClassification = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\trainedClassificationModels\pooling_after_bilstm_2conv_relu_classification.mat"; 
thresholdClassification = .38; % Threshold for determining if something should be rejected. Should be determined based on ROC.

% If the curve is classified as bad, do you want to attempt to trim it so
% that you get a good curve?
ATTEMPT_TO_TRIM_CURVE_TO_FIND_GOOD = 0;
% If yes, what is the minimum amount of depth to consider?
MIN_DEPTH_FOR_GOOD_CLASSIFICATION = 500; %nm



%% AFM Tip Parameters - No need to input spring constant; it is read from the experimental data
% https://www.nanoandmore.com/AFM-Probe-hq-xsc11-hard-al-bs - pattern
% https://www.nanoandmore.com/AFM-Probe-PNP-TR
%  normal = not high aspect, ie, not the one for smiti's micropatterns
% Tip propertie           s based on specific models:

R = 4500;              % Rob's tissue spherical indentation. 4.5 uM, or 4500 nM
% R = 40;              % Radius of curvature for standard silicon nitride tips (nm)
% R = 20;            % Alternate tip radius for specific patterns
% R = 42;            % Tip radius for older Asylum AFM probes
%R = 30;              % Rob's MFS experiments
th = 35 * pi / 180;  % Cone semi-angle for normal silicon nitride probe (radians) (old and new)
% th = 20 * pi / 180; % Alternate semi-angle for specific patterns
%th = 39 * pi / 180; % Rob's MFS
b = R * cos(th);     % Cylindrical radius for modulus calculation
v = 0.5;             % Poisson's ratio for the material


% #################### IMPORTANT %%%%%%%%%%%%%%%%%%%%%%%% !
% The code will automatically assume that the tip is pyramidal, UNLESS you
% specify this parameter below as true (1)
% In this case, the only probe parameter used is R
TIP_IS_SPHERICAL = 1;





% Execute the main AFM analysis script
AFM_POST_JONv6;
end