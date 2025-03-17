%% preprocessTimeRegression_withCaching.m
% Efficient preprocessing of AFM data for regression model timing,
% using file caching to reduce repeated I/O.

clc;
clear;
close all;

%% Parameters
n_points = 2000;
folderPath = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\AFM_data\Tubules";
helperFunctionsFolder = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\helperFunctions";
addpath(helperFunctionsFolder);

numCurvesToProcess = 1000;
numRepetitions = 100;
saveFileName = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\preprocessTimes.mat";

%% Load .mat Files
matFiles = dir(fullfile(folderPath, '*.mat'));
numFiles = length(matFiles);

% Collect all valid curves (store file index, row, col)
validCurves = [];
for i = 1:numFiles
    filePath = fullfile(matFiles(i).folder, matFiles(i).name);
    data = load(filePath, 'indentInfoMap');
    infoIndent = data.indentInfoMap;
    [rows, cols] = find(infoIndent == "Accepted");
    validCurves = [validCurves; repmat(i, length(rows), 1), rows, cols];
end

%% Process random curves repeatedly and record processing times
preprocessingTimes = zeros(numRepetitions, 1);

for iter = 1:numRepetitions
    fprintf('Processing iteration %d/%d...\n', iter, numRepetitions);
    
    % Randomly select curves for this iteration.
    selectedIndices = randperm(size(validCurves,1), numCurvesToProcess);
    selectedCurves = validCurves(selectedIndices, :);
    
    % Build a cache for files that will be used in this iteration.
    % Load each file only once.
    fileCache = cell(numFiles, 1);
    uniqueFileIndices = unique(selectedCurves(:,1));
    for idx = 1:length(uniqueFileIndices)
        fileIdx = uniqueFileIndices(idx);
        filePath = fullfile(matFiles(fileIdx).folder, matFiles(fileIdx).name);
        fileCache{fileIdx} = load(filePath, 'ExtDefl_Matrix', 'Ext_Matrix', 'CP_Matrix');
    end
    
    tic;
    parfor curveIdx = 1:numCurvesToProcess
        fileIdx = selectedCurves(curveIdx, 1);
        row = selectedCurves(curveIdx, 2);
        col = selectedCurves(curveIdx, 3);
        
        % Use cached data instead of loading the file each time.
        data = fileCache{fileIdx};
        deflection = data.ExtDefl_Matrix{row, col};
        extension = data.Ext_Matrix{row, col};
        
        % Normalize deflection data
        normalizedDeflection = (deflection - min(deflection)) / (max(deflection) - min(deflection));
        extensionInterp = linspace(0, 1, n_points);
        deflectionInterp = interp1(linspace(0, 1, length(deflection)), normalizedDeflection, extensionInterp, 'linear', 'extrap');
        
        % Feature calculation (simplified)
        % Note: In MATLAB, smoothdata's third parameter for 'movmean' can be given as a
        % fraction (like 0.006) or a window length. Here we assume it uses the same logic as your original code.
        smoothedDefl = smoothdata(deflectionInterp, 'movmean', 0.006);
        normalizedDefl = (smoothedDefl - min(smoothedDefl)) / (max(smoothedDefl) - min(smoothedDefl));
        
        % (Optional: store features if needed)
    end
    elapsedTime = toc;
    preprocessingTimes(iter) = elapsedTime;
    fprintf('Iteration %d completed in %.2f seconds.\n', iter, elapsedTime);
end

%% Save preprocessing times
save(saveFileName, 'preprocessingTimes');
fprintf('All preprocessing times saved to "%s".\n', saveFileName);
