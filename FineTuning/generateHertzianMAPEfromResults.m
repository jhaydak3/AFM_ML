%% calculate_HertzianModulus_MAPE_fromFineTuning.m
% Compute Hertzian-modulus MAPE for each experiment stored in spherical_test.mat
% and summarise it as a function of training-set size.
%
% Requirements:
%   • 'spherical_test.mat'   – produced by fineTuneEvaluationFunctionOfN.m
%   • the SAME pre-processed data .mat file used for fine-tuning
%   • helperFunctions folder on the MATLAB path (must contain calculateModuli.m)


%% -------------------- User settings ---------------------------------- %%
resultsFile            = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\FineTuning\spherical_test.mat";
preprocessedDataFile   = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_spherical_tissue_5000.mat";
helperFunctionsFolder  = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\helperFunctions";   % contains calculateModuli.m
indentationDepth_nm    = 500;      % depth at which modulus is evaluated (unchanged from earlier scripts)
newResultsFile         = "spherical_test_withModulusMAPE.mat";  % will be created/over-written
rng(1337);                         % reproducibility (for any stochastic helper code)
isTipSpherical = 1;

%% -------------------- Initialise ------------------------------------- %%
clc; close all;
fprintf('Calculating Hertzian-modulus MAPE for fine-tuning results …\n\n');

addpath(helperFunctionsFolder);

if ~exist(resultsFile,"file")
    error("Cannot find results file: %s",resultsFile);
end
if ~exist(preprocessedDataFile,"file")
    error("Cannot find pre-processed data file: %s",preprocessedDataFile);
end

%% -------------------- Load data -------------------------------------- %%
fprintf('Loading fine-tuning results from "%s" …\n',resultsFile);
load(resultsFile, "results", "holdoutIndices");          % struct array & common hold-out indices

fprintf('Loading pre-processed dataset from "%s" …\n',preprocessedDataFile);
data = load(preprocessedDataFile, ...
            "rawExt","rawDefl","minExtValues","maxExtValues", ...
            "b","th","R","v","spring_constant");
%------------------------------------------------------------------------%

nIter        = numel(results);           % total bootstrap × training-size experiments
hertzMAPE    = nan(nIter,1);             % pre-allocate
fprintf('→ %d total experiments detected in results struct.\n',nIter);

%% -------------------- Per-experiment modulus & MAPE ------------------ %%
for k = 1:nIter
    % Retrieve CP predictions & truth (normalised), and the fixed hold-out indices
    YPredHold = results(k).YPredHoldout;   % [nHoldOut×1]
    YTrueHold = results(k).YHoldout;       % [nHoldOut×1]

    % Calculate moduli (this handles CP → indentation conversion internally)
    try
        [HertzActual, HertzPredicted] = calculateModuli( ...
            data.rawExt, data.rawDefl, ...
            YTrueHold, YPredHold, ...
            holdoutIndices, ...
            data.minExtValues, data.maxExtValues, ...
            data.b, data.th, data.R, data.v, data.spring_constant, ...
            indentationDepth_nm, isTipSpherical);

        % Mean Absolute Percent Error (skip NaNs from failed fits)
        hertzMAPE(k)             = mean(abs((HertzPredicted - HertzActual) ./ HertzActual) * 100, ...
                                       'omitnan');
        results(k).hertzMAPE_kPa = hertzMAPE(k);  %#ok<SAGROW> store in struct
    catch ME
        warning("Iter %d: calculateModuli failed (%s). Setting MAPE=NaN.",k,ME.message);
        results(k).hertzMAPE_kPa = NaN;           %#ok<SAGROW>
    end
end

%% -------------------- Aggregate by training-set size ----------------- %%
allSizes      = [results.curveTrainSize];         % vector
uniqueSizes   = unique(allSizes,'sorted');
meanMAPEbyN   = arrayfun(@(n) mean(hertzMAPE(allSizes==n),'omitnan'), uniqueSizes);

fprintf('\n===== Hertzian Modulus – Mean Absolute Percent Error =====\n');
fprintf('Training curves (N)\tMAPE (%%)\n');
for i = 1:numel(uniqueSizes)
    fprintf('%4d\t\t\t%.2f\n', uniqueSizes(i), meanMAPEbyN(i));
end
fprintf('==========================================================\n');

%% -------------------- Dot-plot of all bootstrap replicates ----------- %%
%% -------------------- Jitter-dot + error-bar plot -------------------- %%
fontName     = 'Arial';
fontSize     = 16;
markerSize   = 100;   % matches original square-dot plot
jitterAmount = 2;     % same horizontal jitter as before

figure('Name','Hertzian Modulus MAPE vs Number of Training Curves', ...
       'NumberTitle','off');
hold on;

for i = 1:numel(uniqueSizes)
    currentSize = uniqueSizes(i);

    % indices for this training size
    idxGroup = find(allSizes == currentSize);
    if isempty(idxGroup), continue; end

    mapeVals = hertzMAPE(idxGroup);

    % jittered x-coordinates so dots don’t stack
    xVals = currentSize + (rand(size(mapeVals)) - 0.5) * 2 * jitterAmount;

    % individual bootstrap dots
    scatter(xVals, mapeVals, markerSize, 'filled', ...
            'MarkerFaceColor',[.4 .4 .4], 'Marker','s');

    % mean ± SD error bar
    meanVal = mean(mapeVals, 'omitnan');
    stdVal  = std(mapeVals,  'omitnan');
    errorbar(currentSize, meanVal, stdVal, ...
             'ko', 'LineWidth',1.5, 'CapSize',10, 'MarkerFaceColor','k');
end

% ----- cosmetics ------------------------------------------------------ %
xlabel('Number of curves used for fine-tuning', ...
       'FontName',fontName, 'FontSize',fontSize);
ylabel('Hertzian modulus MAPE (%)', ...
       'FontName',fontName, 'FontSize',fontSize);
set(gca, 'FontName',fontName, 'FontSize',fontSize);
grid on; box on;

xlim([min(uniqueSizes)-10, max(uniqueSizes)+10]);  % pad edges like original plot
% ylim([0 100]);   % set if you want fixed Y limits

hold off;


%% -------------------- Save enriched results -------------------------- %%
fprintf('\nSaving augmented results (with hertzMAPE_kPa field) to "%s" …\n',newResultsFile);
save(newResultsFile, "results", "hertzMAPE", "uniqueSizes", "meanMAPEbyN", "-v7.3");
fprintf('Done.\n');
