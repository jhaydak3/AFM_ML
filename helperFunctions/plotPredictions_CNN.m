function plotPredictions_CNN(results, k)
%PLOTPREDICTIONS_CNN
% Plots Predicted vs Actual Contact Points and Modulus for all folds.
%
% Inputs:
%   results - Structure array containing predictions and actuals for each fold
%   k       - Number of folds

    colors = lines(k); % Generate distinct colors for each fold

    % Create a directory to save figures if it doesn't exist
    figuresDir = 'Evaluation_Figures';
    if ~exist(figuresDir, 'dir')
        mkdir(figuresDir);
    end

    %% -------------------- Calculate Global Limits -------------------- %%
    % CP (Normalized)
    [minCPNorm, maxCPNorm] = calculateGlobalLimits(results, 'YTest', 'YPredTest');
    
    % CP (nm)
    [minCPNm, maxCPNm] = calculateGlobalLimits(results, 'YTestNm', 'YPredTestNm');
    
    % Hertzian Modulus
    [minHertz, maxHertz] = calculateGlobalLimits(results, 'HertzianModulusActual_test', 'HertzianModulusPredicted_test');
    
    % 500 nm Modulus
    [minModulus500, maxModulus500] = calculateGlobalLimits(results, 'Modulus500nmActual_test', 'Modulus500nmPredicted_test');

    %% -------------------- Scatter Plot: CP (Normalized) Predicted vs Actual -------------------- %%
    figure('Name', 'CP (Normalized) Predicted vs Actual', 'NumberTitle', 'off');
    subplotRows = ceil(k / 2);
    subplotCols = 2;
    for fold = 1:k
        subplot(subplotRows, subplotCols, fold);
        scatter(results(fold).YTest, results(fold).YPredTest, 36, colors(fold, :), 'filled');
        hold on;
        % Plot y=x line
        plot([minCPNorm, maxCPNorm], [minCPNorm, maxCPNorm], 'k--', 'LineWidth', 1.5);
        xlabel('Actual CP (Normalized)');
        ylabel('Predicted CP (Normalized)');
        xlim([minCPNorm, maxCPNorm]);
        ylim([minCPNorm, maxCPNorm]);
        title(sprintf('Fold %d', fold));
        grid on;
        hold off;
    end
    sgtitle('Predicted vs Actual CP (Normalized) per Fold');
    savefig(gcf, fullfile(figuresDir, 'CP_Normalized_Predicted_vs_Actual_Subplots.fig'));

    %% -------------------- Scatter Plot: CP (nm) Predicted vs Actual -------------------- %%
    figure('Name', 'CP (nm) Predicted vs Actual', 'NumberTitle', 'off');
    for fold = 1:k
        subplot(subplotRows, subplotCols, fold);
        scatter(results(fold).YTestNm, results(fold).YPredTestNm, 36, colors(fold, :), 'filled');
        hold on;
        % Plot y=x line
        plot([minCPNm, maxCPNm], [minCPNm, maxCPNm], 'k--', 'LineWidth', 1.5);
        xlabel('Actual CP (nm)');
        ylabel('Predicted CP (nm)');
        xlim([minCPNm, maxCPNm]);
        ylim([minCPNm, maxCPNm]);
        title(sprintf('Fold %d', fold));
        grid on;
        hold off;
    end
    sgtitle('Predicted vs Actual CP (nm) per Fold');
    savefig(gcf, fullfile(figuresDir, 'CP_nm_Predicted_vs_Actual_Subplots.fig'));

    %% -------------------- Scatter Plot: Hertzian Modulus Predicted vs Actual -------------------- %%
    figure('Name', 'Hertzian Modulus Predicted vs Actual', 'NumberTitle', 'off');
    for fold = 1:k
        subplot(subplotRows, subplotCols, fold);
        scatter(results(fold).HertzianModulusActual_test, results(fold).HertzianModulusPredicted_test, 36, colors(fold, :), 'filled');
        hold on;
        % Plot y=x line
        plot([minHertz, maxHertz], [minHertz, maxHertz], 'k--', 'LineWidth', 1.5);
        xlabel('Actual Hertzian Modulus (kPa)');
        ylabel('Predicted Hertzian Modulus (kPa)');
        xlim([minHertz, maxHertz]);
        ylim([minHertz, maxHertz]);
        title(sprintf('Fold %d', fold));
        grid on;
        hold off;
    end
    sgtitle('Predicted vs Actual Hertzian Modulus per Fold');
    savefig(gcf, fullfile(figuresDir, 'Hertzian_Modulus_Predicted_vs_Actual_Subplots.fig'));

    %% -------------------- Scatter Plot: 500 nm Modulus Predicted vs Actual -------------------- %%
    figure('Name', '500 nm Modulus Predicted vs Actual', 'NumberTitle', 'off');
    for fold = 1:k
        subplot(subplotRows, subplotCols, fold);
        scatter(results(fold).Modulus500nmActual_test, results(fold).Modulus500nmPredicted_test, 36, colors(fold, :), 'filled');
        hold on;
        % Plot y=x line
        plot([minModulus500, maxModulus500], [minModulus500, maxModulus500], 'k--', 'LineWidth', 1.5);
        xlabel('Actual 500 nm Modulus (kPa)');
        ylabel('Predicted 500 nm Modulus (kPa)');
        xlim([minModulus500, maxModulus500]);
        ylim([minModulus500, maxModulus500]);
        title(sprintf('Fold %d', fold));
        grid on;
        hold off;
    end
    sgtitle('Predicted vs Actual 500 nm Modulus per Fold');
    savefig(gcf, fullfile(figuresDir, '500nm_Modulus_Predicted_vs_Actual_Subplots.fig'));
end

function [minVal, maxVal] = calculateGlobalLimits(results, actualField, predictedField)
% Calculate global min and max values across all folds
    allActual = arrayfun(@(res) res.(actualField), results, 'UniformOutput', false);
    allPredicted = arrayfun(@(res) res.(predictedField), results, 'UniformOutput', false);
    combinedData = [vertcat(allActual{:}); vertcat(allPredicted{:})];
    minVal = min(combinedData);
    maxVal = max(combinedData);
end
