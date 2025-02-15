% consolidatedPreprocess.m
% Comprehensive script for preprocessing AFM data and extracting features for regression

%% Clear Environment
clear;
clc;
close all;

%% Define Parameters
n_points = 2000;               % Number of points for interpolation
folderPath = [
    % "C:\Users\MrBes\Documents\MATLAB\Jon_AFM_Code\version4\Training\Tubules"
    % "C:\Users\MrBes\Documents\MATLAB\Jon_AFM_Code\version4\Training\All_Cancer_Lines"
    %"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\AFM_data\Everything_Jan26"
    % "C:\Users\MrBes\Documents\MATLAB\Jon_AFM_Code\version4\Training\HEPG4"
    % "C:\Users\MrBes\Documents\MATLAB\Jon_AFM_Code\version4\Training\iPSC_VSMC"
    % "C:\Users\MrBes\Documents\MATLAB\Jon_AFM_Code\version4\Training\LM24"
    % "C:\Users\MrBes\Documents\MATLAB\Jon_AFM_Code\version4\Training\MCF7"
    % "C:\Users\MrBes\Documents\MATLAB\Jon_AFM_Code\version4\Training\MCF10a"
    "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\AFM_data\Podocytes"
    ]; % Path to .mat files

preAllocationSize = 10000;     % Initial preallocation size
thresholdFactor = 0.5;         % Factor to determine peak threshold
hertzFrontRemoveAmount = 100;  % Initial depth (in nm) to ignore when calculating fitted Hertz modulus
indentationDepth = 500;        % Indentation depth (nm) to calculate the pointwise modulus. Usually use 500 nm.



% Specify the file name for saving
savedFileName = [
    % "regression_processed_files\processed_features_for_regression_tubules.mat"
    % "regression_processed_files\processed_features_for_regression_All_Cancer_Lines.mat"
    %"regression_processed_files\processed_features_for_regression_All.mat"
    % "regression_processed_files\processed_features_for_regression_HEPG4.mat"
    % "regression_processed_files\processed_features_for_regression_iPSC_VSMC.mat"
    % "regression_processed_files\processed_features_for_regression_LM24.mat"
    % "regression_processed_files\processed_features_for_regression_MCF7.mat"
    % "regression_processed_files\processed_features_for_regression_MCF10a.mat"
    "regression_processed_files\processed_features_for_regression_podocytes.mat"



    ];

addpath("C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\helperFunctions")

%% Start the loop iterating over folders.
numFolders = length(folderPath);
for p = 1:numFolders
    % Allocate variables.
    processedDefl_cell = cell(1, preAllocationSize);     % To store deflection interpolated data per file
    processedExt_cell = cell(1, preAllocationSize);      % To store extension interpolated data per file
    rawDefl_cell = cell(1, preAllocationSize);           % To store raw deflection data per file
    rawDeflInterp_cell = cell(1, preAllocationSize);     % To store raw (interpolated) deflection data per file
    rawExt_cell = cell(1, preAllocationSize);            % To store raw extension data per file
    originalCPIndices_cell = cell(1, preAllocationSize); % To store original CP indices per file
    minDeflValues_cell = cell(1, preAllocationSize);     % To store min deflection per file
    maxDeflValues_cell = cell(1, preAllocationSize);     % To store max deflection per file
    minExtValues_cell = cell(1, preAllocationSize);      % To store min extension per file
    maxExtValues_cell = cell(1, preAllocationSize);      % To store max extension per file
    normalizedCPValues_cell = cell(1, preAllocationSize);% To store normalized CP values per file
    fileIndices_cell = cell(1, preAllocationSize);       % To store file names per file
    modulusHertz_cell = cell(1, preAllocationSize);      % To store Hertz modulus per file
    modulus500_cell = cell(1, preAllocationSize);        % To store 500 nm modulus per file
    fileRow_cell = cell(1,preAllocationSize);            % To store the indentation row
    fileCol_cell = cell(1,preAllocationSize);            % To store the indentation col
    goodOrBad_cell = cell(1, preAllocationSize);         % To store labels: 1 for Good, 0 for Bad


    % **New Storage Variables for Parameters**
    R_cell = cell(1, preAllocationSize);                  % To store R per curve
    v_cell = cell(1, preAllocationSize);                  % To store v per curve
    th_cell = cell(1, preAllocationSize);                 % To store th per curve
    b_cell = cell(1, preAllocationSize);                  % To store b per curve
    spring_constant_cell = cell(1, preAllocationSize);    % To store spring_constant per curve

    %% Load All .mat Files
    matFiles = dir(fullfile(folderPath(p), '*.mat'));
    numFiles = length(matFiles);

    %% Initialize Parallel Pool (Optional)
    % Uncomment the following lines if you want to explicitly start a parallel pool
    % pool = gcp('nocreate');
    % if isempty(pool)
    %     parpool; % Starts the default parallel pool
    % end

    %% Parallel Processing of .mat Files

    parfor i = 1:numFiles
        % Initialize temporary variables for each file
        temp_processedDefl = [];
        temp_processedExt = [];
        temp_rawDefl = {};
        temp_rawDeflInterp = [];
        temp_rawExt = {};
        temp_originalCPIndices = [];
        temp_minDeflValues = [];
        temp_maxDeflValues = [];
        temp_minExtValues = [];
        temp_maxExtValues = [];
        temp_normalizedCPValues = [];
        temp_fileIndices = {};
        temp_fileRow = [];
        temp_fileCol = [];
        temp_modulusHertz = [];
        temp_modulus500 = [];
        temp_goodOrBad = [];


        % **New Temporary Variables for Parameters**
        temp_R = [];
        temp_v = [];
        temp_th = [];
        temp_b = [];
        temp_spring_constant = [];

        % Load the .mat file
        filePath = fullfile(folderPath(p), matFiles(i).name);
        data = load(filePath);

        % Extract relevant matrices
        Defl_Matrix = data.ExtDefl_Matrix;
        Ext_Matrix = data.Ext_Matrix;
        CP_Matrix = data.CP_Matrix;
        infoIndent = data.indentInfoMap;


        % Initialize counters for skipped points
        skipCount = 0;
        totalProcessed = 0;


        for row = 1:size(Defl_Matrix, 1)
            for col = 1:size(Defl_Matrix, 2)
                if infoIndent(row,col) ~= "Bad vibes" && infoIndent(row,col) ~= "Ambiguous CP"

                    checkIfGood = infoIndent(row,col);
                    if checkIfGood == "Ambiguous CP" || checkIfGood == "Bad vibes" || checkIfGood == "Probe slip" || checkIfGood == "Too stiff" || checkIfGood == "Precontact skibidi"
                        isGood = 0;
                    elseif checkIfGood == "Accepted"
                        isGood = 1;
                    else
                        error("Value of infoIndent not recognized.")
                    end

                    deflection = Defl_Matrix{row, col};
                    extension = Ext_Matrix{row, col};

                    % Calculate min and max for scaling
                    minDeflection = min(deflection);
                    maxDeflection = max(deflection);
                    minExtension = min(extension);
                    maxExtension = max(extension);
                    try
                        % Normalize data to [0, 1]
                        if (maxDeflection - minDeflection) == 0
                            normalizedDeflection = zeros(size(deflection));
                        else
                            normalizedDeflection = (deflection - minDeflection) / (maxDeflection - minDeflection);
                        end

                        if (maxExtension - minExtension) == 0
                            normalizedExtension = zeros(size(extension));
                        else
                            normalizedExtension = (extension - minExtension) / (maxExtension - minExtension);
                        end



                        % Interpolate to n_points
                        extensionInterp = linspace(0, 1, n_points);
                        deflectionInterp = interp1(normalizedExtension, normalizedDeflection, extensionInterp, 'linear', 'extrap');

                        % Create interpolated raw deflection for standardized scaling later on.
                        extensionRawInterp = linspace(minExtension, maxExtension, n_points);
                        deflectionRawInterp = interp1(extension, deflection, extensionRawInterp, 'linear');

                        % Convert physical CP value to normalized value
                        normalizedCPValue = (CP_Matrix(row, col) - minExtension) / (maxExtension - minExtension);

                        % Find the closest index in raw extension data to the CP value
                        [~, originalCPIndex] = min(abs(extension - CP_Matrix(row, col)));


                        current_R = data.R;
                        current_th = data.th;
                        current_b = data.b;
                        current_v = data.v;
                        current_spring_constant = data.spring_constant;

                        % Calculate Hertz and 500 nm Elastic modulus
                        % D = (z - z0) - (h - h0), where (z0, h0) is CP
                        depth1 = extension(originalCPIndex:end) - extension(originalCPIndex);
                        depth2 = deflection(originalCPIndex:end) - deflection(originalCPIndex);
                        depth = depth1 - depth2;
                        force = depth2 * current_spring_constant;
                        if depth(end) < hertzFrontRemoveAmount
                            thisE = nan;

                        else
                            [thisE, ~] = calc_E_app(depth, force, current_R, current_th, current_b, 'Hertz', 0, hertzFrontRemoveAmount);
                            % Convert modulus to kPa (Hertz model in Pa)
                            thisE = thisE * 1e18 * 1e-9 / 1000;  % Convert from N/m^2 to kPa
                            thisE = thisE .* 2 .* (1 - current_v.^2);    % Convert from E_apparent to E
                        end

                        if depth(end) < indentationDepth
                            thisE500 = nan;
                        else
                            [~, closestIdx] = min(abs(depth-indentationDepth));
                            F500 = force(closestIdx);
                            D500 = depth(closestIdx);
                            thisE500 = calc_E_singlePoint(D500, F500, current_R, current_th, current_b);
                            thisE500 = thisE500 * 1e18 * 1e-9 / 1000;  % Convert from N/m^2 to kPa
                            thisE500 = thisE500 .* 2 .* (1 - current_v.^2);    % Convert from E_apparent to E
                        end



                        % Store processed data in temporary variables
                        temp_processedDefl = [temp_processedDefl, deflectionInterp'];
                        temp_processedExt = [temp_processedExt, extensionInterp'];
                        temp_rawDefl{end+1} = deflection;
                        temp_rawDeflInterp = [temp_rawDeflInterp, deflectionRawInterp'];
                        temp_rawExt{end+1} = extension;
                        temp_originalCPIndices(end+1) = originalCPIndex;
                        temp_minDeflValues(end+1) = minDeflection;
                        temp_maxDeflValues(end+1) = maxDeflection;
                        temp_minExtValues(end+1) = minExtension;
                        temp_maxExtValues(end+1) = maxExtension;
                        temp_normalizedCPValues(end+1) = normalizedCPValue;
                        temp_fileIndices{end+1} = matFiles(i).name;
                        temp_modulusHertz(end+1) = thisE;
                        temp_modulus500(end+1) = thisE500;
                        temp_fileRow(end+1) = row;
                        temp_fileCol(end+1) = col;
                        temp_goodOrBad(end+1) = isGood; % 1 for Good, 0 for Bad


                        % **Store Parameters for Current Curve**
                        temp_R = [temp_R, current_R];
                        temp_v = [temp_v, current_v];
                        temp_th = [temp_th, current_th];
                        temp_b = [temp_b, current_b];
                        temp_spring_constant = [temp_spring_constant, current_spring_constant];

                        totalProcessed = totalProcessed + 1;
                    catch ME
                        fprintf('Skipping (File: %s, Row: %d, Col: %d) due to error: %s\n', ...
                            matFiles(i).name, row, col, ME.message);
                        skipCount = skipCount + 1;
                        continue; % Skip to the next iteration of the loop
                    end
                end
            end
        end

        % After the loops, display a summary
        % fprintf('File "%s" processing completed.\nTotal contact points processed: %d\nTotal contact points skipped: %d\n', ...
        %     matFiles(i).name, totalProcessed, skipCount);

        % Assign processed data to cell arrays
        processedDefl_cell{i} = temp_processedDefl;
        processedExt_cell{i} = temp_processedExt;
        rawDefl_cell{i} = temp_rawDefl;
        rawDeflInterp_cell{i} = temp_rawDeflInterp;
        rawExt_cell{i} = temp_rawExt;
        originalCPIndices_cell{i} = temp_originalCPIndices;
        minDeflValues_cell{i} = temp_minDeflValues;
        maxDeflValues_cell{i} = temp_maxDeflValues;
        minExtValues_cell{i} = temp_minExtValues;
        maxExtValues_cell{i} = temp_maxExtValues;
        normalizedCPValues_cell{i} = temp_normalizedCPValues;
        fileIndices_cell{i} = temp_fileIndices;
        modulusHertz_cell{i} = temp_modulusHertz;
        modulus500_cell{i} = temp_modulus500;
        fileRow_cell{i} = temp_fileRow;
        fileCol_cell{i} = temp_fileCol;
        goodOrBad_cell{i} = temp_goodOrBad;


        % **Assign Parameters to Storage Cell Arrays**
        R_cell{i} = temp_R;
        v_cell{i} = temp_v;
        th_cell{i} = temp_th;
        b_cell{i} = temp_b;
        spring_constant_cell{i} = temp_spring_constant;
    end


    %% Concatenate All Processed Data
    % Initialize counters
    totalCurves = 0;
    for i = 1:numFiles
        totalCurves = totalCurves + size(processedDefl_cell{i}, 2);
    end

    % Preallocate final arrays based on totalCurves
    processedDefl = zeros(n_points, totalCurves);
    processedExt = zeros(n_points, totalCurves);
    rawDefl = cell(1, totalCurves);
    rawDeflInterp = zeros(n_points, totalCurves);
    rawExt = cell(1, totalCurves);
    originalCPIndices = zeros(1, totalCurves);
    minDeflValues = zeros(1, totalCurves);
    maxDeflValues = zeros(1, totalCurves);
    minExtValues = zeros(1, totalCurves);
    maxExtValues = zeros(1, totalCurves);
    normalizedCPValues = zeros(1, totalCurves);
    fileIndices = cell(1, totalCurves);
    fileRow = zeros(1, totalCurves);
    fileCol = zeros(1, totalCurves);
    modulusHertz = zeros(1, totalCurves);
    modulus500 = zeros(1, totalCurves);
    goodOrBad = zeros(1, totalCurves); % New variable for labels


    % **Preallocate Parameter Vectors**
    R = zeros(1, totalCurves);
    v = zeros(1, totalCurves);
    th = zeros(1, totalCurves);
    b = zeros(1, totalCurves);
    spring_constant = zeros(1, totalCurves);

    % Concatenate data
    currentIndex = 0;
    for i = 1:numFiles
        numCurvesInFile = size(processedDefl_cell{i}, 2);
        if numCurvesInFile > 0
            processedDefl(:, currentIndex+1 : currentIndex+numCurvesInFile) = processedDefl_cell{i};
            processedExt(:, currentIndex+1 : currentIndex+numCurvesInFile) = processedExt_cell{i};
            rawDefl(currentIndex+1 : currentIndex+numCurvesInFile) = rawDefl_cell{i};
            rawDeflInterp(:, currentIndex+1 : currentIndex+numCurvesInFile) = rawDeflInterp_cell{i}; % Fixed assignment
            rawExt(currentIndex+1 : currentIndex+numCurvesInFile) = rawExt_cell{i};
            originalCPIndices(currentIndex+1 : currentIndex+numCurvesInFile) = originalCPIndices_cell{i};
            minDeflValues(currentIndex+1 : currentIndex+numCurvesInFile) = minDeflValues_cell{i};
            maxDeflValues(currentIndex+1 : currentIndex+numCurvesInFile) = maxDeflValues_cell{i};
            minExtValues(currentIndex+1 : currentIndex+numCurvesInFile) = minExtValues_cell{i};
            maxExtValues(currentIndex+1 : currentIndex+numCurvesInFile) = maxExtValues_cell{i};
            normalizedCPValues(currentIndex+1 : currentIndex+numCurvesInFile) = normalizedCPValues_cell{i};
            fileIndices(currentIndex+1 : currentIndex+numCurvesInFile) = fileIndices_cell{i};
            fileRow(currentIndex+1 : currentIndex+numCurvesInFile) = fileRow_cell{i};
            fileCol(currentIndex+1 : currentIndex+numCurvesInFile) = fileCol_cell{i};
            modulusHertz(currentIndex+1 : currentIndex+numCurvesInFile) = modulusHertz_cell{i};
            modulus500(currentIndex+1 : currentIndex+numCurvesInFile) = modulus500_cell{i};
            goodOrBad(currentIndex+1 : currentIndex+numCurvesInFile) = goodOrBad_cell{i}; % Assign labels


            % **Concatenate Parameters**
            R(currentIndex+1 : currentIndex+numCurvesInFile) = R_cell{i};
            v(currentIndex+1 : currentIndex+numCurvesInFile) = v_cell{i};
            th(currentIndex+1 : currentIndex+numCurvesInFile) = th_cell{i};
            b(currentIndex+1 : currentIndex+numCurvesInFile) = b_cell{i};
            spring_constant(currentIndex+1 : currentIndex+numCurvesInFile) = spring_constant_cell{i};

            currentIndex = currentIndex + numCurvesInFile;
        end
    end

    %% Trim Preallocated Cell Arrays (Optional)
    % If preAllocationSize was larger than needed, trim the cell arrays
    % Not necessary if exact preallocation was done


    %% Feature Extraction
    fprintf('Beginning feature extraction. \n')
    % Initialize Features
    smoothFactor1 = 0.006;
    smoothFactor2 = 0.005;
    smoothFactor3 = .1;
    intervalRov = 35;

    numCurves = totalCurves;
    numFeatures = 6; % Initial features: deflection, smoothed deflection, normalized derivative, normalized RoV
    X = zeros(numFeatures, n_points, numCurves);
    Y = zeros(numCurves, 1); % Normalized CP values

    parfor i = 1:numCurves
        % Extract the deflection data
        defl = processedDefl(:, i);

        % Smooth deflection curve
        smoothedDefl = smoothdata(defl, 'movmean', 'SmoothingFactor', smoothFactor1); % Example smoothing factor; adjust as needed
        smoothedDefl = normalizeZeroToOne(smoothedDefl);

        % Compute first derivative with eighth-order accuracy
        eighthOrderDeriv = computeFirstDerivativeEighthOrder(smoothedDefl); % Ensure this function is defined
        smoothedEighthOrderDeriv = smoothdata(eighthOrderDeriv, 'movmean', 'SmoothingFactor', smoothFactor2); % Adjust smoothing factor as needed
        normalizedEighthOrderDeriv = normalizeZeroToOne(smoothedEighthOrderDeriv);

        % Compute the second derivative with eigth-order accuracy
        secondDeriv = computeSecondDerivativeEighthOrder(smoothedDefl);
        smoothedSecondDeriv = smoothdata(secondDeriv, 'movmean', 'SmoothingFactor', smoothFactor3); % Adjust smoothing factor as needed
        normalizedSecondDeriv = normalizeZeroToOne(smoothedSecondDeriv);

        % Compute ratio of variance (RoV)
        rov = generateRoV(smoothedDefl, intervalRov); % Example interval; adjust as needed
        normalizedRov = normalizeZeroToOne(rov);

        % Compute local linear fit.
        [slope, Rsq] = localLinearFit(smoothedDefl, intervalRov);
        normalizedSlope = normalizeZeroToOne(slope);
        normalizedRsq = normalizeZeroToOne(Rsq);

        % Compute moving standard deviation.
        movingStd = movstd(smoothedDefl, [floor(intervalRov/2) floor(intervalRov/2)], 'Endpoints', 'shrink');
        normalizedStd = normalizeZeroToOne(movingStd);

        % Compute moving skewness
        movingSkew = movskew(smoothedDefl, intervalRov);
        normalizedSkew = normalizeZeroToOne(movingSkew);

        % Compute moving kurtosis
        movingKurt = movkurt(smoothedDefl, intervalRov);
        normalizedKurt = normalizeZeroToOne(movingKurt);

        % Compute a moving peak count
        movingPeakCount = movcountpeaks(smoothedDefl, intervalRov)';
        normalizedPeakCount = normalizeZeroToOne(movingPeakCount);

        % Compute a moving sum of signal squared
        movingSum = movsum(smoothedDefl.^2, [floor(intervalRov/2) floor(intervalRov/2)], 'Endpoints', 'shrink');
        normalizedSum = normalizeZeroToOne(movingSum);





        % Data Integrity Checks
        if any(isnan(normalizedRov)) || any(isinf(normalizedRov))
            error('NaN or Inf detected in normalized RoV for sample %d', i);
        end

        % Assemble Features
        % Each feature is a row vector of length n_points
        % Features: defl, smoothedDefl, normalizedEighthOrderDeriv, normalizedRov
        tempFeatures = [
            defl';
            smoothedDefl';
            normalizedEighthOrderDeriv';
            %normalizedSecondDeriv';
            normalizedRov';
            normalizedSlope';
            normalizedRsq';
            %normalizedStd'
            %normalizedSkew';
            %normalizedKurt';
            %normalizedPeakCount';
            %normalizedSum';
            ];

        % Data Integrity Checks for Features
        if any(isnan(tempFeatures), 'all') || any(isinf(tempFeatures), 'all')
            error('NaN or Inf detected in features for sample %d', i);
        end

        % Assign to X
        X(:, :, i) = tempFeatures;

        % Assign the normalized CP value to Y
        Y(i) = normalizedCPValues(i); % Ensure 'normalizedCPValues' exists

        % Data Integrity Check for Target
        if isnan(Y(i)) || isinf(Y(i))
            error('NaN or Inf detected in target for sample %d', i);
        end
    end

    %% Save Features, Targets, and Preprocessed Data



    fprintf('Saving preprocessed data and features to "%s"...\n', savedFileName(p));
    save(savedFileName(p), ...
        'processedDefl', 'processedExt', 'rawDefl', 'rawDeflInterp', 'rawExt', ...
        'originalCPIndices', 'minDeflValues', 'maxDeflValues', ...
        'minExtValues', 'maxExtValues', 'normalizedCPValues', ...
        'fileIndices', 'fileRow', 'fileCol', 'modulusHertz', 'modulus500', ...
        'R', 'v', 'th', 'b', 'spring_constant', 'X', 'Y','goodOrBad');
    fprintf('Preprocessed data saved successfully.\n');

    % Summary
    fprintf('Preprocessing and feature extraction completed successfully.\n');
    fprintf('Total curves processed: %d\n', numCurves);
    fprintf('Processed data saved to "%s"\n', savedFileName(p));
    fprintf('Features and targets (normalized from 0 to 1) saved to "%s"\n', savedFileName(p));
end
%% Supporting Functions
% Ensure the following functions are defined and accessible in your MATLAB path:
% - computeFirstDerivativeEighthOrder
% - generateRoV
% - calc_E_app

function normalized = normalizeZeroToOne(x)
% Make sure it's a vector.
x = x(:);

maxX = max(x);
minX = min(x);

normalized = (x - minX) ./ (maxX - minX);
end


function deriv = computeFirstDerivativeEighthOrder(defl)
% Compute the 8th-order symmetric derivative
eighthOrderDeriv = zeros(size(defl));
h = 1;

for j = 5:(length(defl) - 4)
    eighthOrderDeriv(j) = (1/280 * defl(j-4) - 4/105 * defl(j-3) + 1/5 * defl(j-2) ...
        - 4/5 * defl(j-1) + 4/5 * defl(j+1) - 1/5 * defl(j+2) ...
        + 4/105 * defl(j+3) - 1/280 * defl(j+4)) / h;
end

% Handle boundaries by setting derivatives to zero
eighthOrderDeriv(1:4) = 0;
eighthOrderDeriv(end-3:end) = 0;
deriv = eighthOrderDeriv;
end

function deriv = computeSecondDerivativeEighthOrder(defl)
% computeSecondDerivativeEighthOrder
% Computes the second derivative of a deflection curve with 8th-order accuracy
%
% Syntax:
%   deriv = computeSecondDerivativeEighthOrder(defl)
%
% Inputs:
%   defl - Vector containing deflection data points
%
% Outputs:
%   deriv - Vector containing the second derivative of deflection
%
% Description:
%   This function computes the second derivative of a deflection curve using
%   a symmetric finite difference scheme with 8th-order accuracy. A 9-point stencil
%   is employed to achieve the desired accuracy. Boundary points where the stencil
%   cannot be fully applied are set to zero to maintain array size consistency.
%
% Example:
%   deflection = sin(linspace(0, 2*pi, 1000));
%   secondDeriv = computeSecondDerivativeEighthOrder(deflection);

% Ensure input is a column vector for consistency
defl = defl(:);

% Initialize the derivative vector
deriv = zeros(size(defl));

% Define step size (assuming uniform spacing; set to 1 if unknown)
h = 1;

% Define the coefficients for the 9-point stencil second derivative (8th-order accurate)
coeffs = [-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560];

% Length of the deflection data
N = length(defl);

% Compute the second derivative using the 9-point stencil
% Loop from the 5th point to the (N-4)th point to avoid boundary issues
for j = 5:(N-4)
    deriv(j) = (coeffs(1)*defl(j-4) + coeffs(2)*defl(j-3) + coeffs(3)*defl(j-2) + ...
        coeffs(4)*defl(j-1) + coeffs(5)*defl(j) + coeffs(6)*defl(j+1) + ...
        coeffs(7)*defl(j+2) + coeffs(8)*defl(j+3) + coeffs(9)*defl(j+4)) / (h^2);
end

% Handle boundaries by setting the second derivative to zero
deriv(1:4) = 0;
deriv(end-3:end) = 0;
end


function [slope, R2] = localLinearFit(vector, intervalSize)
% localLinearFit Performs a local linear fit on a vector using a sliding window.
%
%   [slope, R2] = localLinearFit(vector, intervalSize)
%
%   Inputs:
%       vector      - A numerical vector (row or column) containing data points.
%       intervalSize- An odd integer specifying the number of points in each sliding window.
%
%   Outputs:
%       slope       - A vector of the same size as 'vector' containing the slope of the local linear fit at each point.
%       R2          - A vector of the same size as 'vector' containing the R² value of the local linear fit at each point.
%
%   Example:
%       x = 1:100;
%       y = 2*x + randn(1,100); % y = 2x with some noise
%       [slope, R2] = localLinearFit(y, 5);
%       plot(x, y, 'b', x, slope, 'r', x, R2, 'g');
%       legend('Data', 'Local Slope', 'Local R^2');

% Input Validation
if nargin < 2
    error('Function requires two inputs: vector and intervalSize.');
end

if ~isvector(vector)
    error('Input "vector" must be a numerical vector.');
end

if ~isnumeric(intervalSize) || ~isscalar(intervalSize) || intervalSize < 2
    error('Input "intervalSize" must be a positive integer greater than or equal to 2.');
end

intervalSize = round(intervalSize); % Ensure intervalSize is integer

% Ensure intervalSize is odd for centering
if mod(intervalSize, 2) == 0
    warning('intervalSize must be an odd integer. Incrementing intervalSize by 1.');
    intervalSize = intervalSize + 1;
end

halfWindow = floor(intervalSize / 2);

% Convert vector to column vector for consistency
vector = vector(:);
numPoints = length(vector);

% Preallocate output vectors
slope = NaN(numPoints, 1);
R2 = NaN(numPoints, 1);

% Perform local linear fit for each point
for i = 1:numPoints
    % Define window boundaries
    windowStart = max(1, i - halfWindow);
    windowEnd = min(numPoints, i + halfWindow);
    windowIndices = windowStart:windowEnd;
    windowSizeActual = length(windowIndices);

    % Ensure there are at least two points to perform linear fit
    if windowSizeActual < 2
        slope(i) = NaN;
        R2(i) = NaN;
        continue;
    end

    % Define x and y for linear fit
    x = windowIndices;
    y = vector(windowIndices);

    % Perform linear regression: y = a*x + b
    p = polyfit(x, y, 1);
    slope(i) = p(1); % Extract slope

    % Compute R^2
    y_fit = polyval(p, x)';
    SS_res = sum((y - y_fit).^2);
    SS_tot = sum((y - mean(y)).^2);
    if SS_tot == 0
        R2(i) = NaN; % Undefined R^2
    else
        R2(i) = 1 - (SS_res / SS_tot);
    end
end

% Convert outputs to the original shape (row or column)
if isrow(vector)
    slope = slope';
    R2 = R2';
end
end

function peakCount = movcountpeaks(data, window)
n = length(data);
peakCount = zeros(size(data));
halfWindow = floor(window / 2);
for i = 1:n
    startIdx = max(1, i - halfWindow);
    endIdx = min(n, i + halfWindow);
    windowData = data(startIdx:endIdx);
    [pks, ~] = findpeaks(windowData);
    peakCount(i) = length(pks);
end
end


function skew = movskew(data, window)
n = length(data);
skew = zeros(size(data));
halfWindow = floor(window / 2);
for i = 1:n
    startIdx = max(1, i - halfWindow);
    endIdx = min(n, i + halfWindow);
    windowData = data(startIdx:endIdx);
    skew(i) = skewness(windowData);
end

end

function kurt = movkurt(data, window)
n = length(data);
kurt = zeros(size(data));
halfWindow = floor(window / 2);
for i = 1:n
    startIdx = max(1, i - halfWindow);
    endIdx = min(n, i + halfWindow);
    windowData = data(startIdx:endIdx);
    kurt(i) = kurtosis(windowData);
end
end

