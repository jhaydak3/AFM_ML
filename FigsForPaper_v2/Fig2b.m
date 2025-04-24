close all
clear
clc

% List of .mat files and model names
modelFiles = {
    "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\evaluation\regression\SNAP\iterative_evaluation_results.mat";
    "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\evaluation\regression\AzelogluCosta2011\piecewise_evaluation_results.mat"; 
    %"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\evaluation\regression\modified_AzelogluCosta\piecewise_evaluation_results.mat", ...
"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\evaluation\regression\CNNs\Sotres2022_original_no_augmentation.mat"; 
   "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\evaluation\regression\CNNs\Sotres2022_ResNet50-1D_no_augmentation.mat"; 
"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\evaluation\regression\CNNs\two_conv_biLSTM_sequence_pooling_relu_no_augmentation.mat"};
%modelNames = {'Linear-Quadratic','Linear-Power','ConvNet-1D', 'ResNet50-1D', 'ConvLSTM-1D'};
modelNames = {'SNAP','Linear-Quadratic','ConvNet-1D', 'ResNet50-1D', 'COBRA'};

% Check if the number of models and files match
if length(modelFiles) ~= length(modelNames)
    error('Number of model files and names must match.');
end

% Set up colors and markers for the models
numModels = length(modelFiles);
colors = lines(numModels); % Generate a colormap with 'numModels' distinct colors
%markers = {'+', 'o', 'x', '*', '.'}; % Marker styles
markers = {'o','+', 'x', '*', '.'}; % Marker styles

%% Figure 1: Contact Point Predictions vs. Actual Contact Points (Logarithmic Axes)
figure;
hold on;
allErrorsContactPoints = cell(1, numModels);
for m = 1:numModels
    % Load the model's data
    load(modelFiles{m});
    allActualContactPoints = [];
    allPredictedContactPoints = [];

    % Handle data format differences
    if exist('results', 'var') % Old format
        allActualContactPoints = [];
        allPredictedContactPoints = [];

        for i = 1:length(results)
            allActualContactPoints = [allActualContactPoints; results(i).YTestNm];
            allPredictedContactPoints = [allPredictedContactPoints; results(i).YPredTestNm];
        end
    elseif exist('trueInNm', 'var') && exist('predInNm', 'var') % New format
        allActualContactPoints = trueInNm;
        allPredictedContactPoints = predInNm;
    end

    % Calculate errors and store for histogram
    errorsContactPoints = allPredictedContactPoints - allActualContactPoints;
    allErrorsContactPoints{m} = errorsContactPoints;

    % Plot aggregated data for the current model
    scatter(allActualContactPoints, allPredictedContactPoints, 50, 'MarkerEdgeColor', colors(m,:), ...
        'Marker', markers{mod(m-1, length(markers))+1}, 'DisplayName', modelNames{m});
end
hold on;

% Customize the plot
set(gca, 'XScale', 'log', 'YScale', 'log'); % Set axes to logarithmic scale
axis tight; % Adjust axis limits
xLimits = get(gca, 'XLim'); % Get tight axis limits
yLimits = get(gca, 'YLim');
commonLimits = [min([xLimits(1), yLimits(1)]), max([xLimits(2), yLimits(2)])]; % Ensure equal limits
set(gca, 'XLim', commonLimits, 'YLim', commonLimits); % Apply common limits
xlabel('Actual Contact Points (nm)');
ylabel('Predicted Contact Points (nm)');
title('Contact Points: Predictions vs. Actual');
legend('show', 'Location', 'best');
grid on;

% Add identity line (y = x)
xIdentity = logspace(log10(commonLimits(1)), log10(commonLimits(2)), 100); % Generate points on the identity line
plot(xIdentity, xIdentity, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Identity Line');

lb = 4 * 1e1;
ub = 2 * 1e4;
xlim([lb ub])
ylim([lb ub])
clear results Y_true Y_Pred_piecewise
%% Figure 1B: Contact Point Predictions vs. Actual (normalized)
figure;
hold on;
allErrorsContactPoints = cell(1, numModels);
for m = 1:numModels

    clear results Y_true Y_Pred_piecewise
    % Load the model's data
    load(modelFiles{m});
    allActualContactPoints = [];
    allPredictedContactPoints = [];

    % Handle data format differences
    if exist('results', 'var') % Old format
        allActualContactPoints = [];
        allPredictedContactPoints = [];

        for i = 1:length(results)
            allActualContactPoints = [allActualContactPoints; results(i).YTest];
            allPredictedContactPoints = [allPredictedContactPoints; results(i).YPredTest];
        end
    elseif exist('Y_true', 'var') && exist('YPred_piecewise', 'var') % New format
        allActualContactPoints = Y_true;
        allPredictedContactPoints = YPred_piecewise;
    end

    % Calculate errors and store for histogram
    errorsContactPoints = allPredictedContactPoints - allActualContactPoints;
    allErrorsContactPoints{m} = errorsContactPoints;

    % Plot aggregated data for the current model
    scatter(allActualContactPoints, allPredictedContactPoints, 50, 'MarkerEdgeColor', colors(m,:), ...
        'Marker', markers{mod(m-1, length(markers))+1}, 'DisplayName', modelNames{m});
end
hold on;

% Customize the plot
set(gca, 'XScale', 'linear', 'YScale', 'linear'); % Set axes to logarithmic scale
axis tight; % Adjust axis limits
xLimits = get(gca, 'XLim'); % Get tight axis limits
yLimits = get(gca, 'YLim');
commonLimits = [min([xLimits(1), yLimits(1)]), max([xLimits(2), yLimits(2)])]; % Ensure equal limits
set(gca, 'XLim', commonLimits, 'YLim', commonLimits); % Apply common limits
xlabel('Actual Contact Points (normalized)');
ylabel('Predicted Contact Points (normalized)');
title('Contact Points: Predictions vs. Actual');
legend('show', 'Location', 'best');
grid on;

% Add identity line (y = x)
xIdentity = logspace(log10(commonLimits(1)), log10(commonLimits(2)), 100); % Generate points on the identity line
plot(xIdentity, xIdentity, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Identity Line');

lb = .2;
ub = 1.05;
xlim([lb ub])
ylim([lb ub])

%% Figure 2: Hertzian Modulus Actual vs. Predicted (Linear Axes)
figure;
hold on;
allPercentErrorsHertz = cell(1, numModels);
clear results
clear E_hertz_actual
clear E_hertz_pred

for m = 1:numModels
    % Load the model's data
    load(modelFiles{m});
    allActualHertz = [];
    allPredictedHertz = [];

    % Handle data format differences
    if exist('results', 'var') % Old format
        allActualContactPoints = [];
        allPredictedContactPoints = [];

        for i = 1:length(results)
            allActualHertz = [allActualHertz; results(i).HertzianModulusActual_test];
            allPredictedHertz = [allPredictedHertz; results(i).HertzianModulusPredicted_test];
        end
    elseif exist('E_hertz_actual', 'var') && exist('E_hertz_pred', 'var') % New format
        allActualHertz = E_hertz_actual;
        allPredictedHertz = E_hertz_pred;
    end

    % Calculate percent errors and store for histogram
    percentErrorsHertz = 100 * (allPredictedHertz - allActualHertz) ./ allActualHertz;
    allPercentErrorsHertz{m} = percentErrorsHertz;

    % Plot aggregated data for the current model
    scatter(allActualHertz, allPredictedHertz, 50, 'MarkerEdgeColor', colors(m,:), ...
        'Marker', markers{mod(m-1, length(markers))+1}, 'DisplayName', modelNames{m});

end
hold on;

% Customize the plot
axis tight; % Adjust axis limits
xlabel('Actual Hertzian Modulus (kPa)');
ylabel('Predicted Hertzian Modulus (kPa)');
title('Hertzian Modulus: Predictions vs. Actual');
legend('show', 'Location', 'best');
grid on;
set(gca,'Yscale','log')
set(gca,'Xscale','log')

% Add identity line (y = x)
lb = .95e-1;
ub = 3e2;
xlim([lb ub])
ylim([lb ub])
xIdentity = linspace(lb, ub, 100); % Generate points on the identity line
plot(xIdentity, xIdentity, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Identity Line');

%% Calculate common bin edges for Contact Point Errors
allContactErrors = vertcat(allErrorsContactPoints{:}); % Combine all errors
numBinsContact = 2000; % Set the number of bins
contactBinEdges = linspace(min(allContactErrors), max(allContactErrors), numBinsContact + 1);

%% Calculate common bin edges for Hertzian Modulus Percent Errors
allHertzPercentErrors = vertcat(allPercentErrorsHertz{:}); % Combine all percent errors
numBinsHertzPercent = 2500; % Set the number of bins
hertzPercentBinEdges = linspace(min(allHertzPercentErrors), max(allHertzPercentErrors), numBinsHertzPercent + 1);
hertzPercentBinEdges = -500:1.5:500;

%% Figure 3: Histograms of Contact Point Errors (Unified Bins)


% Overlay KDE on Histogram for Contact Point Errors
figure;
hold on;
for m = 1:numModels
   % Plot histogram with transparency
    histogram(allErrorsContactPoints{m}, 'FaceColor', colors(m,:), ...
        'FaceAlpha', 0.5, 'DisplayName', modelNames{m}, ...
        'Normalization', 'probability', 'BinEdges', contactBinEdges, ...
        'EdgeColor', colors(m,:), 'EdgeAlpha', 0.5);

    % Compute and overlay KDE
     [f, xi] = ksdensity(allErrorsContactPoints{m}, -1:.0001:1);
     plot(xi, f*.00068, 'Color', colors(m,:), 'LineWidth', 2, 'HandleVisibility', 'off'); % Exclude KDE from legend
end
hold off;

% Customize plot
xlabel('Normalized Contact Point Error');
ylabel('Proportion');
title('Normalized Contact Point Errors Histogram');
legend('show', 'Location', 'best'); % Legend for histograms only
grid on;
xlim([-.1 .1]);

xticks([-.2:.025:.2])

xlbub = .05;
xlim([-xlbub xlbub])



%% Figure 4: Histograms of Hertzian Modulus Percent Errors (Unified Bins)
figure;
hold on;
scaleFactor = 1.73;
for m = 1:numModels
    % Plot histogram with transparency
    histogram(allPercentErrorsHertz{m}, 'FaceColor', colors(m,:), ...
        'FaceAlpha', 0.5, 'DisplayName', modelNames{m}, ...
        'Normalization', 'probability', 'BinEdges', hertzPercentBinEdges, ...
        'EdgeColor', colors(m,:), 'EdgeAlpha', 0.5);

    % Compute and overlay KDE
    thisScaleFactor = scaleFactor;
    % if m == 5
    %     thisScaleFactor = 1.7;
    % end
    [f, xi] = ksdensity(allPercentErrorsHertz{m}, -50:.005:50); % Adjust range and bin width for KDE
    plot(xi, f * thisScaleFactor, 'Color', colors(m,:), 'LineWidth', 2, 'HandleVisibility', 'off'); % Exclude KDE from legend
end
hold off;

% Customize plot
xlabel('Hertzian Modulus Percent Error (%)');
ylabel('Proportion');
title('Hertzian Modulus Percent Errors Histogram');
legend('show', 'Location', 'best'); % Legend for histograms only
grid on;

% Adjust x-axis limits for better visualization
xlim([-50 50]); % Example range, adjust as needed

% Set x-ticks
xticks(-50:25:50);

% Set x-tick labels (optional, but can reinforce proper labeling)
xticklabels(arrayfun(@num2str, -50:25:50, 'UniformOutput', false));