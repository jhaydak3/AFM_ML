function Fig1_new(useSavedResultsIfAvailable)
    % SWEEP_CONTACT_POINT_ERROR
    %
    % Demonstrates how small errors in identifying the contact point (CP)
    % can lead to large errors in the calculated moduli (both at 500 nm
    % and Hertzian). Additionally classifies each curve’s error into
    % user-defined categories, computes the fraction of curves in each
    % category, and produces stacked area plots across offsets.
    %

    % If the user doesn't supply "useSavedResultsIfAvailable", set default:
    if nargin < 1
        useSavedResultsIfAvailable = true;
    end
    
    clc; close all;  % Clear command window and close figures

    %% ------------------ Configuration ------------------ %%
    % Path to the same data file used previously
    dataFile ="C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\training\regression_processed_files\processed_features_for_regression_All.mat";

    % Range of offsets (in nm) to apply to the "actual" contact point
    rangeToSweep = -300:50:300;  % e.g. [-300, -200, -100, 0, 100, 200, 300]

    % Indentation depth (nm) for the 500 nm modulus calculation
    indentationDepth_nm = 500;

    % Whether to save the resulting figures
    saveFigure = true;
    outputFigNameError_Hertz = 'CP_Offset_ErrorPlot_Hertz.fig';
    outputFigNameError_500nm = 'CP_Offset_ErrorPlot_500nm.fig';
    outputFigNameCats_500 = 'CP_Offset_StackedArea_500nm.fig';
    outputFigNameCats_Hertz = 'CP_Offset_StackedArea_Hertz.fig';

    % -------------------- Error Categories (modifiable) -------------------- %
    categoryEdges = [0, 5, 10, 20, 40, Inf];
    categoryLabels = {'\le5%', '5-10%', '10-20%', '20-40%', '>40%'};

    % This is where we will store results from the sweep so we don't
    % have to re-run expensive calculations multiple times:
    analysisDataFile = 'CP_Offset_AnalysisResults.mat';

    %% --------- Attempt to Load Precomputed Results (if desired) --------- %%
    if useSavedResultsIfAvailable && isfile(analysisDataFile)
        % If user asked to reuse saved results and file exists, load from it
        load(analysisDataFile, ...
             'rangeToSweep', 'meanAbsPctErr_500nm', 'stdAbsPctErr_500nm', ...
             'meanAbsPctErr_Hertz', 'stdAbsPctErr_Hertz', ...
             'fracByCat_500', 'fracByCat_Hertz', ...
             'categoryEdges', 'categoryLabels');
        fprintf('Loaded previously saved sweep results from: %s\n', analysisDataFile);

    else
        %% ------------------ Load the Data (and re-run analysis) ------------------ %%
        if ~isfile(dataFile)
            error('Data file not found: %s', dataFile);
        end
        dataStruct = load(dataFile);

        % Check for required variables
        requiredVars = ["Y", "maxExtValues", "minExtValues", ...
                        "rawExt", "rawDefl", "R", "b", "th", ...
                        "spring_constant", "v"];
        for rv = requiredVars
            if ~isfield(dataStruct, rv)
                error('Missing required variable "%s" in dataFile.', rv);
            end
        end

        % Extract relevant data
        Y_actual           = dataStruct.Y;              % [numCurves x 1] normalized CP
        maxExtValues       = dataStruct.maxExtValues';  % [1 x numCurves]
        minExtValues       = dataStruct.minExtValues';  % [1 x numCurves]
        rawExt             = dataStruct.rawExt;         % {1 x numCurves}
        rawDefl            = dataStruct.rawDefl;        % {1 x numCurves}
        R_all              = dataStruct.R;              % [1 x numCurves], tip radius (nm)
        b_all              = dataStruct.b;              % [1 x numCurves]
        th_all             = dataStruct.th;             % [1 x numCurves]
        k_all              = dataStruct.spring_constant;% [1 x numCurves] (N/m)
        v_all              = dataStruct.v;              % [1 x numCurves] Poisson ratio

        numCurves          = length(Y_actual);

        % Create a logical index that includes all curves
        dataIdx = 1:numCurves;

        % Convert "actual" normalized CP to actual CP in nm
        CP_actual_nm = Y_actual .* (maxExtValues - minExtValues) + minExtValues;

        %% ------------------ Allocate Arrays ------------------ %%
        numOffsets = length(rangeToSweep);

        meanAbsPctErr_500nm  = zeros(numOffsets, 1);
        stdAbsPctErr_500nm   = zeros(numOffsets, 1);
        meanAbsPctErr_Hertz  = zeros(numOffsets, 1);
        stdAbsPctErr_Hertz   = zeros(numOffsets, 1);

        nCats = length(categoryEdges) - 1;
        fracByCat_500   = zeros(numOffsets, nCats);
        fracByCat_Hertz = zeros(numOffsets, nCats);

        %% ------------------ Reference Moduli (Actual CP) ------------------ %%
        % Calculate the "actual" moduli for each curve (using true CP).
        YPred_dummy = Y_actual;  % same as Y_actual
        [HertzianModulusActual, ~, Modulus500nmActual, ~] = ...
            calculateModuli(rawExt, rawDefl, ...
                            Y_actual, YPred_dummy, ...
                            dataIdx, ...
                            minExtValues, maxExtValues, ...
                            b_all, th_all, R_all, v_all, ...
                            k_all, indentationDepth_nm);

        %% ------------------ Sweep Over Offsets ------------------ %%
        for oIdx = 1:numOffsets
            offset_nm = rangeToSweep(oIdx);  % current offset in nm

            if offset_nm == 0
                % Per your prior instruction: "When the offset is 0, assume errors = 0"
                meanAbsPctErr_500nm(oIdx) = 0;
                stdAbsPctErr_500nm(oIdx)  = 0;
                meanAbsPctErr_Hertz(oIdx) = 0;
                stdAbsPctErr_Hertz(oIdx)  = 0;

                % Put all curves into the first bin (<=5%)
                fracByCat_500(oIdx, :)   = [1, zeros(1, nCats-1)];
                fracByCat_Hertz(oIdx, :) = [1, zeros(1, nCats-1)];
                continue;
            end

            % Build the "predicted" CP in nm by adding the offset
            CP_predicted_nm = CP_actual_nm + offset_nm;

            % Convert predicted CPs in nm back into normalized units
            Y_pred = zeros(numCurves, 1);
            for iCurve = 1:numCurves
                denom = (maxExtValues(iCurve) - minExtValues(iCurve));
                if denom ~= 0
                    Y_pred(iCurve) = (CP_predicted_nm(iCurve) - minExtValues(iCurve)) / denom;
                else
                    Y_pred(iCurve) = NaN;
                end
            end

            % Call calculateModuli() with these predicted CPs
            [~, HertzianModulusPred, ~, Modulus500nmPred] = ...
                calculateModuli(rawExt, rawDefl, ...
                                Y_actual, Y_pred, ...
                                dataIdx, ...
                                minExtValues, maxExtValues, ...
                                b_all, th_all, R_all, v_all, ...
                                k_all, indentationDepth_nm);

            %% ------------ Compute Absolute Percent Errors ------------ %%
            % 500 nm modulus
            valid500Idx = ~isnan(Modulus500nmActual) & ...
                          ~isnan(Modulus500nmPred)   & ...
                           (Modulus500nmActual ~= 0);
            absPctErr_500 = nan(numCurves, 1);
            absPctErr_500(valid500Idx) = ...
                100 * abs(Modulus500nmPred(valid500Idx) - Modulus500nmActual(valid500Idx)) ...
                    ./ abs(Modulus500nmActual(valid500Idx));

            meanAbsPctErr_500nm(oIdx) = mean(absPctErr_500(valid500Idx), 'omitnan');
            stdAbsPctErr_500nm(oIdx)  = std(absPctErr_500(valid500Idx), 'omitnan');

            % Hertz modulus
            validHertzIdx = ~isnan(HertzianModulusActual) & ...
                            ~isnan(HertzianModulusPred)   & ...
                             (HertzianModulusActual ~= 0);
            absPctErr_Hertz = nan(numCurves, 1);
            absPctErr_Hertz(validHertzIdx) = ...
                100 * abs(HertzianModulusPred(validHertzIdx) - HertzianModulusActual(validHertzIdx)) ...
                    ./ abs(HertzianModulusActual(validHertzIdx));

            meanAbsPctErr_Hertz(oIdx) = mean(absPctErr_Hertz(validHertzIdx), 'omitnan');
            stdAbsPctErr_Hertz(oIdx)  = std(absPctErr_Hertz(validHertzIdx), 'omitnan');

            %% ------ Classify Each Curve's Error into Categories ------ %%
            fracByCat_500(oIdx,:) = ...
                categorizeErrors(absPctErr_500, categoryEdges);
            fracByCat_Hertz(oIdx,:) = ...
                categorizeErrors(absPctErr_Hertz, categoryEdges);
        end

        %% -------------- Save the computed results to a MAT file -------------- %%
        save(analysisDataFile, ...
             'rangeToSweep', ...
             'meanAbsPctErr_500nm', 'stdAbsPctErr_500nm', ...
             'meanAbsPctErr_Hertz', 'stdAbsPctErr_Hertz', ...
             'fracByCat_500', 'fracByCat_Hertz', ...
             'categoryEdges', 'categoryLabels');
        fprintf('Analysis finished. Results saved to: %s\n', analysisDataFile);
    end

    %% ------------------ Plot Mean ± Std Errors ------------------ %%
    figure('Name','Errors vs Contact-Point Offset, Hertzian Modulus','NumberTitle','off');
    hold on; grid on;
    errorbar(rangeToSweep, meanAbsPctErr_Hertz, stdAbsPctErr_Hertz, ...
             'o-','LineWidth',1.5, 'MarkerSize',6, 'Color',[0.8500 0.3250 0.0980]); % red
    xlabel('Offset from True Contact Point (nm)');
    ylabel('Absolute Percent Error (%)');
    legend({'Hertzian Modulus'}, 'Location','best');

    if saveFigure
        savefig(gcf, outputFigNameError_Hertz);
        fprintf('Figure saved to "%s".\n', outputFigNameError_Hertz);
    end

    figure('Name','Errors vs Contact-Point Offset, 500nm Point','NumberTitle','off');
    hold on; grid on;
    errorbar(rangeToSweep, meanAbsPctErr_500nm, stdAbsPctErr_500nm, ...
        'o-','LineWidth',1.5, 'MarkerSize',6, 'Color',[0 0.4470 0.7410]);  % blue
    xlabel('Offset from True Contact Point (nm)');
    ylabel('Absolute Percent Error (%)');
    legend({'Pointwise Modulus at 500 nm'}, 'Location','best');

    if saveFigure
        savefig(gcf, outputFigNameError_500nm);
        fprintf('Figure saved to "%s".\n', outputFigNameError_500nm);
    end

    % Optional: One figure with both
    figure('Name','Errors vs Contact-Point Offset (Combined)','NumberTitle','off');
    hold on; grid on;
    errorbar(rangeToSweep, meanAbsPctErr_Hertz, stdAbsPctErr_Hertz, ...
        'o-','LineWidth',1.5, 'MarkerSize',6, 'Color',[0 0 0]); % red
    errorbar(rangeToSweep, meanAbsPctErr_500nm, stdAbsPctErr_500nm, ...
        'o--','LineWidth',1.5, 'MarkerSize',6, 'Color',[.4 .4 .4]);  % blue
    xlabel('Offset from True Contact Point (nm)');
    ylabel('Absolute Percent Error (%)');
    legend({'Hertzian Modulus','500 nm Pointwise Modulus'}, 'Location','best');

    %% ------------------ Stacked Area Plots (Categories) ------------------ %%
    % fracByCat_500 and fracByCat_Hertz are in decimal fraction form, multiply by 100.
    figure('Name','Stacked Area - Error Categories (500 nm)','NumberTitle','off');
    area(rangeToSweep, fracByCat_500 * 100, 'LineStyle', 'none');
    colormap jet; 
    ylabel('Percent of Curves');
    xlabel('Offset from True Contact Point (nm)');
    title('Distribution of Error Categories - 500 nm Modulus');
    legend(categoryLabels, 'Location','best');
    grid on;

    if saveFigure
        savefig(gcf, outputFigNameCats_500);
        fprintf('Figure saved to "%s".\n', outputFigNameCats_500);
    end

    figure('Name','Stacked Area - Error Categories (Hertz)','NumberTitle','off');
    area(rangeToSweep, fracByCat_Hertz * 100, 'LineStyle', 'none');
    colormap jet;  
    ylabel('Percent of Curves');
    xlabel('Offset from True Contact Point (nm)');
    title('Distribution of Error Categories - Hertzian Modulus');
    legend(categoryLabels, 'Location','best');
    grid on;

    if saveFigure
        savefig(gcf, outputFigNameCats_Hertz);
        fprintf('Figure saved to "%s".\n', outputFigNameCats_Hertz);
    end

    fprintf('Sweep completed. Plots generated.\n');

end


%% ------------------ Helper Function ------------------ %%
function fracByCat = categorizeErrors(errorArray, edges)
    % CATEGORIZEERRORS
    %   errorArray: Nx1 vector of percent errors (NaNs included).
    %   edges:      e.g. [0, 5, 10, 20, 40, Inf].
    %
    % Returns a 1xC row vector (C=length(edges)-1) of the fraction
    % of samples that fall into each category.
    %
    %   - All NaNs go into the last category (highest error).
    %   - edges must be sorted with Inf at the end.

    nCats = length(edges) - 1;
    N = length(errorArray);

    counts = zeros(1, nCats);

    for i = 1:N
        val = errorArray(i);
        if isnan(val)
            % Put in last bin
            counts(nCats) = counts(nCats) + 1;
        else
            % edges = [0, 5, 10, 20, 40, Inf]
            % We want edges(cIdx) < val <= edges(cIdx+1)
            placedFlag = false;
            for cIdx = 1:nCats
                if (val > edges(cIdx)) && (val <= edges(cIdx+1))
                    counts(cIdx) = counts(cIdx) + 1;
                    placedFlag = true;
                    break;
                end
            end
            if ~placedFlag
                counts(nCats) = counts(nCats) + 1;
            end
        end
    end

    fracByCat = counts / N;
end
