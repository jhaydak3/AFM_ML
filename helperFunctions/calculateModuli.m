function [HertzianModulusActual, HertzianModulusPredicted, Modulus500nmActual, Modulus500nmPredicted] = calculateModuli(rawExt, rawDefl, YActual, YPred, ...
    dataIdx, minExtValues, maxExtValues, b, th, R, v, spring_constant, indentationDepth_nm, ...
    isTipSpherical)
%CALCULATEMODULI
% Calculates Hertzian and 500 nm moduli for both actual and predicted Contact Points (CPs).
%
% Inputs:
%   rawExt               - {1 x totalCurves} cell array of raw extension data (nm)
%   rawDefl              - {1 x totalCurves} cell array of raw deflection data (nm)
%   YActual              - [numSamples x 1] normalized CP values (0 to 1)
%   YPred                - [numSamples x 1] predicted normalized CP values (0 to 1)
%   dataIdx              - [1 x totalCurves] logical index for the current dataset (train or test)
%   minExtValues         - [1 x totalCurves] min extension values for each curve (nm)
%   maxExtValues         - [1 x totalCurves] max extension values for each curve (nm)
%   b                    - [1 x totalCurves] physical parameter 'b' for each curve
%   th                   - [1 x totalCurves] thickness parameter for each curve
%   R                    - [1 x totalCurves] radius parameter for each curve (nm)
%   v                    - [1 x totalCurves] Poisson's ratio for each curve
%   spring_constant      - [1 x totalCurves] spring constant for each curve (N/nm)
%   indentationDepth_nm  - Scalar value specifying the indentation depth (500 nm)
%
% Outputs:
%   HertzianModulusActual     - [numSamples x 1] Actual Hertzian Modulus (kPa)
%   HertzianModulusPredicted - [numSamples x 1] Predicted Hertzian Modulus (kPa)
%   Modulus500nmActual        - [numSamples x 1] Actual 500 nm Modulus (kPa)
%   Modulus500nmPredicted    - [numSamples x 1] Predicted 500 nm Modulus (kPa)

% ---------- handle optional argument ----------
if nargin < 14          % only 13 inputs supplied → no flag passed
    isTipSpherical = false;
end
% ----------------------------------------------

% Initialize output arrays
numSamples = length(dataIdx);
HertzianModulusActual = NaN(numSamples,1);
HertzianModulusPredicted = NaN(numSamples,1);
Modulus500nmActual = NaN(numSamples,1);
Modulus500nmPredicted = NaN(numSamples,1);

% Define HertzFrontRemoveAmount (assumed as 100 nm, adjust if needed)
HertzFrontRemoveAmount = 100; % nm
% Define the minimum depth required for a curve to be considered in
% Hertzian modulus calculation
min_Depth_Hertzian = 100; %nm

for i = 1:numSamples
    % Find the current curve index
    curveIdx = dataIdx(i);         % Use the i-th element from idxAll

    % Retrieve raw data for the current curve
    extension = rawExt{curveIdx}(:);   % [N x 1] vector
    deflection = rawDefl{curveIdx}(:); % [N x 1] vector

    % Ensure column vectors
    if size(extension, 2) > 1
        extension = extension.';
    end
    if size(deflection, 2) > 1
        deflection = deflection.';
    end

    % Retrieve physical parameters for the current curve
    current_R = R(curveIdx);
    current_th = th(curveIdx);
    current_b = b(curveIdx);
    current_v = v(curveIdx);
    current_spring_constant = spring_constant(curveIdx);

    %% ----------------- Calculate Depth and Force for Predicted CP ----------------- %%
    % Calculate Predicted CP in nm
    predictedCP_nm = YPred(i) * (maxExtValues(curveIdx) - minExtValues(curveIdx)) + minExtValues(curveIdx);

    % Find index closest to predicted CP in extension_actual
    [~, idx_pred] = min(abs(extension - predictedCP_nm));

    % Retrieve z0_pred and h0_pred
    z0_pred = extension(idx_pred);
    h0_pred = deflection(idx_pred);

    % Calculate depth and force arrays for predicted CP
    depth1_pred = extension(idx_pred:end) - z0_pred;          % [N - idx_pred +1 x 1]
    depth2_pred = deflection(idx_pred:end) - h0_pred;        % [N - idx_pred +1 x 1]
    depth_pred = depth1_pred - depth2_pred;                  % [N - idx_pred +1 x 1]
    force_pred = depth2_pred * current_spring_constant;      % [N - idx_pred +1 x 1]

    % Remove any NaN or Inf values
    validIndices_pred = ~isnan(depth_pred) & ~isnan(force_pred) & ...
        ~isinf(depth_pred) & ~isinf(force_pred);
    depth_pred = depth_pred(validIndices_pred);
    force_pred = force_pred(validIndices_pred);

    % Ensure sufficient data points
    if depth_pred(end) >= indentationDepth_nm
        % Calculate pointwise modulus using 'pointwise' mode with try-catch
        try
            [~, closest_idx_actual] = min(abs(depth_pred - indentationDepth_nm));
            F500 = force_pred(closest_idx_actual);
            D500 = depth_pred(closest_idx_actual);
            E500_apparent = calc_E_singlePoint(D500, F500, current_R, current_th, current_b, isTipSpherical);
            % Convert to kPa
            E500_temp  = E500_apparent * 1e18 * 1e-9 / 1000;  % nN/(nM^2) to kPa
            % Adjust based on Poisson's ratio
            E500 = E500_temp * 2 * (1 - current_v^2);

            % [E_pointwise_pred_temp] = calc_E_app(depth_pred, force_pred, ...
            %     current_R, current_th, current_b, ...
            %     'pointwise', 0, 0);
            % % Convert to kPa
            % E_pointwise_pred_temp_kPa = E_pointwise_pred_temp * 1e18 * 1e-9 / 1000;  % nN/(nM^2) to kPa
            %
            % % Adjust based on Poisson's ratio
            % E_pointwise_pred_kPa = E_pointwise_pred_temp_kPa * 2 * (1 - current_v^2);
            %
            % % Find the index closest to indentationDepth_nm (500 nm)
            % [~, closest_idx_pred] = min(abs(depth_pred - indentationDepth_nm));
            %
            % % Assign the modulus at the closest depth
            % if ~isempty(closest_idx_pred)
            %     Modulus500nmPredicted(i) = E_pointwise_pred_kPa(closest_idx_pred);
            % end
            Modulus500nmPredicted(i) = E500;
        catch
            Modulus500nmPredicted(i) = NaN;
        end
    end

    %% ----------------- Calculate Depth and Force for Actual CP ----------------- %%
    % Calculate Actual CP in nm
    actualCP_nm = YActual(i) * (maxExtValues(curveIdx) - minExtValues(curveIdx)) + minExtValues(curveIdx);

    % Find index closest to actual CP in extension_actual
    [~, idx_actual] = min(abs(extension - actualCP_nm));

    % Retrieve z0_actual and h0_actual
    z0_actual = extension(idx_actual);
    h0_actual = deflection(idx_actual);

    % Calculate depth and force arrays for actual CP
    depth1_actual = extension(idx_actual:end) - z0_actual;        % [N - idx_actual +1 x 1]
    depth2_actual = deflection(idx_actual:end) - h0_actual;      % [N - idx_actual +1 x 1]
    depth_actual = depth1_actual - depth2_actual;                % [N - idx_actual +1 x 1]
    force_actual = depth2_actual * current_spring_constant;      % [N - idx_actual +1 x 1]

    % Remove any NaN or Inf values
    validIndices_actual = ~isnan(depth_actual) & ~isnan(force_actual) & ...
        ~isinf(depth_actual) & ~isinf(force_actual);
    depth_actual = depth_actual(validIndices_actual);
    force_actual = force_actual(validIndices_actual);

    % Ensure sufficient data points
    if depth_actual(end) >= indentationDepth_nm
        % Calculate pointwise modulus using 'pointwise' mode with try-catch
        try
            [~, closest_idx_actual] = min(abs(depth_actual - indentationDepth_nm));
            F500 = force_actual(closest_idx_actual);
            D500 = depth_actual(closest_idx_actual);
            E500_apparent = calc_E_singlePoint(D500, F500, current_R, current_th, current_b, isTipSpherical);
            % Convert to kPa
            E500_temp  = E500_apparent * 1e18 * 1e-9 / 1000;  % % nN/(nM^2) to kPa
            % Adjust based on Poisson's ratio
            E500 = E500_temp * 2 * (1 - current_v^2);
            % [E_pointwise_actual_temp] = calc_E_app(depth_actual, force_actual, ...
            %     current_R, current_th, current_b, ...
            %     'pointwise', 0, 0);
            % % Convert to kPa
            % E_pointwise_actual_temp_kPa = E_pointwise_actual_temp * 1e18 * 1e-9 / 1000;  % % nN/(nM^2) to kPa
            %
            % % Adjust based on Poisson's ratio
            % E_pointwise_actual_kPa = E_pointwise_actual_temp_kPa * 2 * (1 - current_v^2);
            %
            % % Find the index closest to indentationDepth_nm (500 nm)
            % [~, closest_idx_actual] = min(abs(depth_actual - indentationDepth_nm));
            %
            % % Assign the modulus at the closest depth
            % if ~isempty(closest_idx_actual)
            %     Modulus500nmActual(i) = E_pointwise_actual_kPa(closest_idx_actual);
            % end
            Modulus500nmActual(i) = E500;
        catch
            Modulus500nmActual(i) = NaN;
        end
    end

    %% ----------------- Hertzian Modulus Calculation ----------------- %%
    % Calculate Hertzian Modulus using 'Hertz' mode with try-catch
    if depth_actual(end) > min_Depth_Hertzian
        try
            [E_hertz_val_actual] = calc_E_app(depth_actual, force_actual, ...
                current_R, current_th, current_b, ...
                'Hertz', 0, HertzFrontRemoveAmount, isTipSpherical);
            HertzianModulusActual(i) = E_hertz_val_actual * 1e18 * 1e-9 / 1000;  % N/m^2 to kPa
            % Adjust based on Poisson's ratio
            HertzianModulusActual(i) = HertzianModulusActual(i) * 2 * (1 - current_v^2);
        catch
            HertzianModulusActual(i) = NaN;
        end
    end
    if depth_pred(end) > min_Depth_Hertzian
        try
            [E_hertz_val_pred] = calc_E_app(depth_pred, force_pred, ...
                current_R, current_th, current_b, ...
                'Hertz', 0, HertzFrontRemoveAmount, isTipSpherical);
            HertzianModulusPredicted(i) = E_hertz_val_pred * 1e18 * 1e-9 / 1000;  % N/m^2 to kPa
            % Adjust based on Poisson's ratio
            HertzianModulusPredicted(i) = HertzianModulusPredicted(i) * 2 * (1 - current_v^2);
        catch
            HertzianModulusPredicted(i) = NaN;
        end
    end
end
end