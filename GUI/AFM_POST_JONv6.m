%% Script to Process HDF5 Files from Asylum's .ARDF Converter (Single-Loop Version)
% This script processes the h5 file produced by Asylum's .ARDF converter.
% It reads the data, extracts force curves, calculates contact points, and estimates
% elastic modulus using a piecewise fit with normalization & interpolation.
%
% (Single-loop version to ease parallelization via parfor)
%
% Authors: J.H., E. U. A.

% Disable warnings temporarily
warning off;

%% File Loading and Initialization

% Load the HDF5 file info
info = h5info(h5_file_loc);
info2 = h5info(h5_file_loc, '/ForceMap');

% Get dataset labels for force curves
mdata = info2.Groups.Datasets;
FCnames = {mdata.Name}';  % Force Curve names

% Exclude the last dataset ('segment') and convert to string array
FCnames_str = string(FCnames(1:end-1));
sndx = strfind(FCnames_str, ':');  % Find the ':' separator
sndx = [sndx{:}]';  % Convert to column vector

% Pre-allocate arrays for row and column indices
nData = length(FCnames_str);
rndx = zeros(nData, 1);
cndx = zeros(nData, 1);

% Parse row and column indices from force curve names
for j = 1:nData
    this_string = char(FCnames_str(j));
    rndx(j) = str2double(this_string(1:sndx(j)-1));  % Row index (zero-indexed)
    cndx(j) = str2double(this_string(sndx(j)+1:end));  % Column index (zero-indexed)
end

% Determine total number of rows and columns (add 1 because indices are zero-indexed)
nRows = max(rndx) + 1;
nCols = max(cndx) + 1;

%% Read Spring Constant from Metadata

metadata = info.Attributes.Value;
metadata = strsplit(metadata, '\n');
metadata = string(metadata);
TF = contains(metadata, 'SpringConstant: 0.');
metadata = metadata(TF);
string_w_sc = char(metadata(1));
pndx = strfind(string_w_sc, '.');
sc = string_w_sc(pndx-1:end);
spring_constant = str2double(sc);  % in N/m (or nN/nM)

%% Initialize Matrices for Storing Results

E_Matrix = zeros(nRows, nCols);  % Elastic modulus
CP_Matrix = zeros(nRows, nCols); % Contact points
HertzianModulus_Matrix = zeros(nRows,nCols); % Hertzian modulus
AcceptRejectMap = true(nRows,nCols); % This is initialized as true.


if SAVE_OPT == 1
    F_Matrix = cell(nRows, nCols);
    D_Matrix = cell(nRows, nCols);
    Ext_Matrix = cell(nRows, nCols);     % Extension data for fitting
    ExtDefl_Matrix = cell(nRows, nCols); % Deflection data
    PWE_Matrix = cell(nRows, nCols);     % Pointwise modulus

end

%% Process Force Curves Using a Single Loop

% Read the segment data (last dataset) indicating where curves switch from extension to retraction
segmentdata = h5read(h5_file_loc, ['/ForceMap/0/' FCnames{end}]);

totalCurves = nRows * nCols;
tic;  % Start timer

% Use a single loop over all force curves. For each k, determine row and column indices.
for k = 1:totalCurves
    % Compute row (i2) and column (i) indices from k (assuming column-major order)
    i2 = mod(k-1, nRows) + 1;         % Row index (1-based)
    i = floor((k-1) / nRows) + 1;       % Column index (1-based)

    % Generate string identifier (using zero-indexing as in the file)
    curID = string([num2str(i2-1) ':' num2str(i-1)]);

    % Check if the current force curve exists in FCnames_str
    if ~any(strcmp(FCnames_str, curID))
        continue;  % Skip this (row, col) if no force curve exists
    end

    % Read data for the current force curve
    dataPath = ['/ForceMap/0/' num2str(i2-1) ':' num2str(i-1)];
    data = h5read(h5_file_loc, dataPath);

    % Extract data columns: raw, deflection (converted to nm), Z sensor (converted to nm)
    raw = data(:, 1);
    defl = data(:, 2) / 1e-9;
    Zsnsr = data(:, 3) / 1e-9;

    % Get extension and retraction indices for the current (i2, i)
    extndx = segmentdata(1, i, i2);  % End of extension
    retndx = segmentdata(2, i, i2);  % End of retraction

    % Plot raw data if enabled
    if PLOT_OPT == 1
        figure;
        plot(Zsnsr(1:extndx), defl(1:extndx));
        title('Raw Data: Extension vs Deflection');
        xlabel('Extension (nm)');
        ylabel('Deflection (nm)');
    end

    % Define fitting region (using all points up to extndx)
    start_idx = 1;
    fit_ext = Zsnsr(start_idx:extndx);
    fit_defl = defl(start_idx:extndx);

    % Check for non-monotonic extension data
    if any(diff(fit_ext) < 0)
        warning('Non-monotonic extension data at row %d col %d.', i2, i);
    end

    %% Predict whether this is a good quality curve.

    if PREDICT_QUALITY_OPT == 1
        predictedClassification = predict_NN_GUI_classification(fit_ext, fit_defl, networkModelClassification);
        % If the first value is above the threshold, it should be rejected.
        if predictedClassification >= thresholdClassification
            AcceptRejectMap(i2,i) = false;
        else
            AcceptRejectMap(i2,i) = true;
        end

    end

    %% Find the Point of Contact using Piecewise Fit with Normalization & Interpolation
    extMin = min(fit_ext);
    extMax = max(fit_ext);
    defMin = min(fit_defl);
    defMax = max(fit_defl);

    if (extMax - extMin) < eps || (defMax - defMin) < eps
        fprintf('Near-constant curve at row %d col %d. Skipping contact fit.\n', i2, i);
        minxcndx = start_idx;  % fallback
    else
        if CONTACT_METHOD_OPT == 1
            % Define hyperparameters for the piecewise fit function
            dataFraction = 0.3;
            lb = [0, -0.5, 2, 0, 0, 0, 1];
            ub = [1, 0.5, 2, 500, 500, 0, 1];

            % Call external function (assumed to be implemented) to get predicted contact
            predictedCP = linearquadratic_GUI(fit_ext, fit_defl, dataFraction, lb, ub);
        elseif CONTACT_METHOD_OPT == 2
            maxIter = 20;
            offsetFraction = 0.3;
            deflThreshold = 2;  % nm
            deflFitRange = [0, 15];  % nm
            plotDebug = false;  % Change to true for debugging
            predictedCP = SNAP_GUI(fit_ext, fit_defl, spring_constant, v, th, ...
                maxIter, offsetFraction, deflThreshold, deflFitRange, plotDebug, b, R);
        elseif CONTACT_METHOD_OPT == 3
            predictedCP = predict_NN_GUI(fit_ext,fit_defl,networkModel);
        else
            error('Contact method not implemented')
        end

        if isnan(predictedCP)
            minxcndx = start_idx;  % fallback
        else
            [~, minxcndx_fit] = min(abs(fit_ext - predictedCP));
            minxcndx = minxcndx_fit + start_idx - 1;
            if PLOT_OPT == 1
                figure('Name','Method 1: Piecewise Fit','NumberTitle','off');
                hold on;
                plot(fit_ext, fit_defl, 'b-*','DisplayName','Raw Data');

                % Optional: Evaluate the piecewise model on a fine grid (requires pBest)
                xFine = linspace(0,1,400);
                yFine = piecewiseFun_v3(pBest, xFine);  % Assumes pBest is available
                extPlot = xFine*(extMax - extMin) + extMin;
                deflPlot = yFine*(defMax - defMin) + defMin;
                plot(extPlot, deflPlot, 'r-','LineWidth',1.5,'DisplayName','Piecewise Fit');
                plot(fit_ext(minxcndx_fit), fit_defl(minxcndx_fit), 'ko','MarkerSize',8,'DisplayName','Contact Point');
                xlabel('Z sensor / Extension (nm)');
                ylabel('Deflection (nm)');
                title(sprintf('Piecewise Fit (row=%d, col=%d)', i2, i));
                legend('Location','best');
                grid on;
            end
        end
    end

    % Store contact point (in Z sensor units)
    CP_Matrix(i2, i) = Zsnsr(minxcndx);

    %% Calculate Elastic Modulus
    def_val = defl(minxcndx:extndx) - defl(minxcndx);  % Indentation (h - h0)
    D = Zsnsr(minxcndx:extndx) - Zsnsr(minxcndx);        % Z sensor shift (z - z0)
    D = D - def_val;                                    % Correct indentation depth
    F = def_val .* spring_constant;                     % Force vector (using Hooke's law)

    % Calculate apparent elastic modulus (pointwise calculation)
    [E_app, regimeChange] = calc_E_app(D, F, R, th, b, 'pointwise', PLOT_OPT);
    % Convert to kPa: first to Pa then to kPa
    E_app = E_app * 1e18 * 1e-9;  % Now in Pa
    E_app = E_app / 1000;         % Now in kPa
    E = E_app .* 2 .* (1 - v^2);  % Correct for Poisson's ratio

    % Store elastic modulus (last value of E)
    E_Matrix(i2, i) = E(end);

    % Calculate the fitted Hertzian Modulus
    try
        E_app = calc_E_app(D, F, R, th, b, 'Hertz', 0, HERTZIAN_FRONT_REMOVE);
        E_app = E_app * 1e18 * 1e-9;  % Conversions remain unchanged
        E_app = E_app / 1000;
        HertzianModulus_Matrix(i2, i) = E_app;
    catch
        HertzianModulus_Matrix(i2, i) = nan;
    end

    %% Save raw data if enabled
    if SAVE_OPT == 1
        F_Matrix{i2, i} = F;
        D_Matrix{i2, i} = D;
        Ext_Matrix{i2, i} = Zsnsr(1:extndx);
        ExtDefl_Matrix{i2, i} = defl(1:extndx);
        PWE_Matrix{i2, i} = real(E);
        % Note: RoV_Matrix and RoVZ_Matrix are not computed in this trimmed version.
    end

    % Optionally display progress (every ~10% of total iterations)
    if mod(k, round(totalCurves/10)) == 0
        fprintf('%.2f percent done processing.\n', (k/totalCurves)*100);
    end
end
toc;

%% Plotting Results

% Ensure no imaginary parts remain
E_Matrix = real(E_Matrix);

if PLOT_OPT == 1
    % Plot elastic modulus map
    figure;
    imagesc(E_Matrix);
    title('Elastic Modulus');
    c = colorbar;
    set(gca, 'FontSize', FontSize, 'YDir', 'normal');
end

% Plot relative height map based on contact points
Height_Matrix = max(CP_Matrix(:)) - CP_Matrix;

if PLOT_OPT == 1
    figure;
    imagesc(Height_Matrix);
    title('Relative Height (CP estimate)');
    c = colorbar;
    set(gca, 'FontSize', FontSize, 'YDir', 'normal');
end

%% Save Results

if SAVE_OPT == 1
    % Ensure SAVE_NAME exists and ends with .mat
    if ~exist('SAVE_NAME','var') || isempty(SAVE_NAME)
        SAVE_NAME = 'results.mat';
    elseif ~endsWith(SAVE_NAME, '.mat')
        SAVE_NAME = [SAVE_NAME '.mat'];
    end

    save(SAVE_NAME, 'E_Matrix', 'F_Matrix', 'D_Matrix', 'CP_Matrix', ...
        'h5_file_loc', 'R', 'th', 'b', ...
         'Ext_Matrix', 'ExtDefl_Matrix', 'Height_Matrix', ...
        'spring_constant', 'v', 'PWE_Matrix', "HertzianModulus_Matrix", ...
        'AcceptRejectMap','CONTACT_METHOD_OPT','PREDICT_QUALITY_OPT', ...
        'thresholdClassification','networkModelClassification', ...
        'networkModel');
end

%% Re-enable Warnings
warning on;
