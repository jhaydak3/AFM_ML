close all
clc

% 3180
plotSingleCurveAndPredictionsHelper(3200)

function plotSingleCurveAndPredictionsHelper(indexToPlot)
    % PLOTSINGLECURVEANDPREDICTIONS  Plots one AFM curve in nm 
    % (extension vs. deflection) plus vertical lines showing:
    %   - Actual contact point (red)
    %   - Predicted contact points from each of the 5 models (in distinct colors).
    %
    % Usage:
    %   plotSingleCurveAndPredictions(1000);
    %
    % NOTE: This script converts from normalized extension/deflection to nm
    % using minExtValues, maxExtValues, minDeflValues, and maxDeflValues.
    % The predicted CPs from the model files (predInNm, YPredTestNm, etc.)
    % are already in nm, so no further conversion is required there.

    % --------------------------------------------------------------
    % 1) Load the main preprocessed data (contains processedDefl, processedExt, etc.)
    % --------------------------------------------------------------
    mainMatFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\training\regression_processed_files\processed_features_for_regression_All.mat";
    load(mainMatFile, ...
         'processedDefl', 'processedExt', 'originalCPIndices', ...
         'maxDeflValues', 'minDeflValues', 'maxExtValues', 'minExtValues', ...
         'rawExt','rawDefl'); 

    % --------------------------------------------------------------
    % 2) Extract the single curve data for the chosen index
    % --------------------------------------------------------------
    numCurves = size(processedExt, 2);
    if indexToPlot < 1 || indexToPlot > numCurves
        error('Index %d is out of range (1 to %d).', indexToPlot, numCurves);
    end

    % Normalized extension/deflection 
    extensionDataNorm  = processedExt(:, indexToPlot);
    deflectionDataNorm = processedDefl(:, indexToPlot);
    extensionRaw = rawExt{indexToPlot};
    deflectionRaw = rawDefl{indexToPlot};

    % Convert them to nm using the corresponding min/max for this curve
    extMin = minExtValues(indexToPlot);
    extMax = maxExtValues(indexToPlot);
    deflMin = minDeflValues(indexToPlot);
    deflMax = maxDeflValues(indexToPlot);

    extensionData_nm = extensionDataNorm * (extMax - extMin) + extMin;
    deflectionData_nm = deflectionDataNorm * (deflMax - deflMin) + deflMin;

    e0 = extensionRaw(1);
    extensionRaw = extensionRaw - e0;
    d0 = min(deflectionRaw);
    deflectionRaw = deflectionRaw-d0;

    % The actual contact point index corresponds to the raw extension.
    cpSampleIndex = originalCPIndices(indexToPlot);

    % Convert that to an x-value in nm:
    actualCP_nm = extensionRaw(cpSampleIndex);

    % --------------------------------------------------------------
    % 3) Set up the figure and plot the curve
    % --------------------------------------------------------------
    figure('Name', sprintf('Example Prediction', indexToPlot), 'Color', 'w'); 
    hold on; grid on;

    % Plot the extension vs. deflection curve in nm
    plot(extensionRaw, deflectionRaw, ...
         'k*', 'LineWidth', 1, 'DisplayName', 'Force Curve','MarkerSize',2);

        plot(NaN, NaN, 'w', 'LineStyle','none', ...
     'DisplayName',' ');

    % Actual CP in red
    xline(actualCP_nm, 'Color', 'k', 'LineWidth', 2, 'LineStyle', '-', ...
          'DisplayName', 'Actual CP', 'Alpha',1);

    % --------------------------------------------------------------
    % 4) Prepare model file paths, names, colors
    % --------------------------------------------------------------
    modelFiles = {
            "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\evaluation\regression\SNAP\iterative_evaluation_results.mat", ...
    "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\evaluation\regression\AzelogluCosta2011\piecewise_evaluation_results.mat", ...
"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\evaluation\regression\CNNs\Sotres2022_original_no_augmentation.mat", ...
      "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\evaluation\regression\CNNs\Sotres2022_ResNet50-1D_no_augmentation.mat", ...
"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\evaluation\regression\CNNs\results_same_epochs\two_conv_LSTM_sequence_pooling_relu_no_augmentation.mat"
    };

    modelNames = { ...
        'SNAP', ...
        'Linear-Quadratic', ...
        'ConvNet-1D', ...
        'ResNet50-1D', ...
        'ConvLSTM-1D' ...
    };

    numModels   = numel(modelFiles);
    modelColors = lines(numModels);
    markers     = { 'o','+', 'x', '*', '.'};  %

    % --------------------------------------------------------------
    % 5) For each model, load it and plot the predicted CP (in nm)
    % --------------------------------------------------------------
    for m = 1:numModels
        thisModelFile = modelFiles{m};
        thisModelName = modelNames{m};

        % Load the variables from the model’s .mat file
        S = load(thisModelFile);

        switch m
            case {1,2}  % The first two have direct [2967 x 1] arrays: predInNm
                if ~isfield(S, 'predInNm')
                    error('Model file %s does not have "predInNm".', thisModelFile);
                end
                predictedCP_nm = S.predInNm(indexToPlot);

            otherwise
                % The last three have 10-fold cross validation in S.results
                if ~isfield(S, 'results')
                    error('Model file %s does not have a "results" struct.', thisModelFile);
                end

                foundFold = 0;
                foldArray = S.results;
                for ff = 1:numel(foldArray)
                    testInds = foldArray(ff).testIndices;
                    if ismember(indexToPlot, testInds)
                        foundFold = ff;
                        break;
                    end
                end

                if foundFold == 0
                    warning('Index %d not found in testIndices for model %s.', ...
                             indexToPlot, thisModelFile);
                    predictedCP_nm = NaN;
                else
                    posInFold      = find(foldArray(foundFold).testIndices == indexToPlot);
                    predictedCP_nm = foldArray(foundFold).YPredTestNm(posInFold);
                end
        end

        % Plot a vertical line for this model’s predicted CP
        xline(predictedCP_nm - e0, ...
              'Color',     modelColors(m,:), ...
              'LineWidth', 1.5, ...
              'LineStyle', '--', ...
              'DisplayName', thisModelName, 'Alpha',1);
    end

    % --------------------------------------------------------------
    % 6) Final figure labeling
    % --------------------------------------------------------------
    xlabel('Extension (nm)');
    ylabel('Deflection (nm)');
    title(sprintf('Example Prediction', indexToPlot), ...
          'FontWeight', 'bold');
    legend('Location', 'best');
    hold off;
end
