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

% metadata = info.Attributes.Value;
% metadata = strsplit(metadata, '\n');
% metadata = string(metadata);
% %TF = contains(metadata, 'SpringConstant: 0.');
% TF = startsWith(metadata, 'SpringConstant: ');
% metadata = metadata(TF);
% string_w_sc = char(metadata(1));
% pndx = strfind(string_w_sc, '.');
% sc = string_w_sc(pndx-1:end);
% spring_constant = str2double(sc);  % in N/m (or nN/nM)

metadata = info.Attributes.Value;
metadata = strsplit(metadata, '\n');
metadata = string(metadata);
TF = startsWith(metadata, 'SpringConstant: ');
springConstantLine = metadata(TF);

if ~isempty(springConstantLine)
    % Use a regex to capture the number following 'SpringConstant: '
    token = regexp(springConstantLine(1), 'SpringConstant:\s*([0-9]*\.?[0-9]+)', 'tokens', 'once');
    if ~isempty(token)
        spring_constant = str2double(token{1});
    else
        error('Could not parse the spring constant from the metadata.');
    end
else
    error('SpringConstant entry not found in metadata.');
end


%% Initialize Matrices for Storing Results

E_Matrix = zeros(nRows, nCols);          % Elastic modulus
CP_Matrix = zeros(nRows, nCols);           % Contact points
HertzianModulus_Matrix = zeros(nRows, nCols); % Hertzian modulus
AcceptRejectMap = true(nRows, nCols);      % Original quality map

% New quality map for the trimmed (clipped) curves
AcceptRejectMapClipped = false(nRows, nCols);

if SAVE_OPT == 1
    F_Matrix = cell(nRows, nCols);
    D_Matrix = cell(nRows, nCols);
    Ext_Matrix = cell(nRows, nCols);      % Extension data for fitting
    ExtDefl_Matrix = cell(nRows, nCols);    % Deflection data (full curve)
    PWE_Matrix = cell(nRows, nCols);        % Pointwise modulus
    % New cell array for clipped deflection curves (the trimmed version)
    ExtDefl_Matrix_Clipped = cell(nRows, nCols);
end

%% Process Force Curves Using a Single Loop

% Read the segment data (last dataset) indicating where curves switch from extension to retraction
segmentdata = h5read(h5_file_loc, ['/ForceMap/0/' FCnames{end}]);

totalCurves = nRows * nCols;
tic;  % Start timer

% Preallocate temporary output cell arrays (length = totalCurves)
out_i2               = cell(totalCurves,1);
out_i                = cell(totalCurves,1);
out_E                = cell(totalCurves,1);
out_CP               = cell(totalCurves,1);
out_Hertzian         = cell(totalCurves,1);
out_Accept           = cell(totalCurves,1);
out_AcceptClipped    = cell(totalCurves,1);
out_F                = cell(totalCurves,1);
out_D                = cell(totalCurves,1);
out_Ext              = cell(totalCurves,1);
out_ExtDefl          = cell(totalCurves,1);
out_ExtDeflClipped   = cell(totalCurves,1);
out_PWE              = cell(totalCurves,1);

parfor k = 1:totalCurves
    % Compute row (i2) and column (i) indices from k (column-major order)
    local_i2 = mod(k-1, nRows) + 1;  % Row index (1-based)
    local_i  = floor((k-1) / nRows) + 1;  % Column index (1-based)
    out_i2{k} = local_i2;
    out_i{k}  = local_i;
    
    % Generate string identifier (using zero-indexing as in the file)
    curID = string([num2str(local_i2-1) ':' num2str(local_i-1)]);
    
    % Check if the current force curve exists in FCnames_str
    if ~any(strcmp(FCnames_str, curID))
        % If not, assign empty outputs for this iteration
        out_E{k}              = NaN;
        out_CP{k}             = NaN;
        out_Hertzian{k}       = NaN;
        out_Accept{k}         = false;
        out_AcceptClipped{k}  = false;
        out_F{k}              = [];
        out_D{k}              = [];
        out_Ext{k}            = [];
        out_ExtDefl{k}        = [];
        out_ExtDeflClipped{k} = [];
        out_PWE{k}            = [];
        continue;
    end
    
    % Read data for the current force curve
    dataPath = ['/ForceMap/0/' num2str(local_i2-1) ':' num2str(local_i-1)];
    data = h5read(h5_file_loc, dataPath);
    
    % Extract data columns: raw, deflection (converted to nm), Z sensor (converted to nm)
    raw    = data(:, 1);
    defl   = data(:, 2) / 1e-9;
    Zsnsr  = data(:, 3) / 1e-9;
    
    % Get extension and retraction indices for the current (local_i2, local_i)
    extndx = segmentdata(1, local_i, local_i2);  % End of extension
    retndx = segmentdata(2, local_i, local_i2);    % End of retraction
    
    % (Optional) Plot raw data if enabled – note: plotting in parfor is not recommended
    if PLOT_OPT == 1
        figure;
        plot(Zsnsr(1:extndx), defl(1:extndx));
        title('Raw Data: Extension vs Deflection');
        xlabel('Extension (nm)');
        ylabel('Deflection (nm)');
    end
    
    % Define fitting region (all points up to extndx)
    start_idx = 1;
    fit_ext   = Zsnsr(start_idx:extndx);
    fit_defl  = defl(start_idx:extndx);
    
    % Check for non-monotonic extension data
    if any(diff(fit_ext) < 0)
        warning('Non-monotonic extension data at row %d col %d.', local_i2, local_i);
    end
    
    % Initialize trimming flag and variables
    trimmingNeeded = false;
    % Predict quality on full (untrimmed) curve
    if PREDICT_QUALITY_OPT == 1
        initialPred = predict_NN_GUI_classification(fit_ext, fit_defl, networkModelClassification);
        if initialPred < thresholdClassification
            local_Accept = true;
            trimmed_ext  = fit_ext;
            trimmed_defl = fit_defl;
            local_AcceptClipped = true;
        else
            local_Accept = false;
            if ATTEMPT_TO_TRIM_CURVE_TO_FIND_GOOD == 1
                trimmingNeeded = true;
            else
                trimmingNeeded = false;
                trimmed_ext  = fit_ext;
                trimmed_defl = fit_defl;
            end
            local_AcceptClipped = false;  % until a good candidate is found
        end
    else
        trimmed_ext  = fit_ext;
        trimmed_defl = fit_defl;
        local_Accept = true;
        local_AcceptClipped = true;
    end
    
    %% Find the Point of Contact using Piecewise Fit with Normalization & Interpolation
    extMin = min(fit_ext);
    extMax = max(fit_ext);
    defMin = min(fit_defl);
    defMax = max(fit_defl);
    
    if (extMax - extMin) < eps || (defMax - defMin) < eps
        fprintf('Near-constant curve at row %d col %d. Skipping contact fit.\n', local_i2, local_i);
        minxcndx = start_idx;  % fallback
    else
        if CONTACT_METHOD_OPT == 1
            dataFraction = 0.3;
            lb = [0, -0.5, 2, 0, 0, 0, 1];
            ub = [1, 0.5, 2, 500, 500, 0, 1];
            predictedCP = linearquadratic_GUI(fit_ext, fit_defl, dataFraction, lb, ub);
        elseif CONTACT_METHOD_OPT == 2
            maxIter = 20;
            offsetFraction = 0.3;
            deflThreshold = 2;  % nm
            deflFitRange = [0, 15];  % nm
            plotDebug = false;
            predictedCP = SNAP_GUI(fit_ext, fit_defl, spring_constant, v, th, ...
                maxIter, offsetFraction, deflThreshold, deflFitRange, plotDebug, b, R);
        elseif CONTACT_METHOD_OPT == 3
            predictedCP = predict_NN_GUI(fit_ext, fit_defl, networkModel);
        else
            error('Contact method not implemented')
        end
        
        if isnan(predictedCP)
            minxcndx = start_idx;
        else
            [~, minxcndx_fit] = min(abs(fit_ext - predictedCP));
            minxcndx = minxcndx_fit + start_idx - 1;
            if PLOT_OPT == 1
                figure('Name','Piecewise Fit','NumberTitle','off');
                hold on;
                plot(fit_ext, fit_defl, 'b-*','DisplayName','Raw Data');
                plot(fit_ext(minxcndx_fit), fit_defl(minxcndx_fit), 'ko','MarkerSize',8,'DisplayName','Contact Point');
                xlabel('Z sensor / Extension (nm)');
                ylabel('Deflection (nm)');
                title(sprintf('Piecewise Fit (row=%d, col=%d)', local_i2, local_i));
                legend('Location','best');
                grid on;
                hold off;
            end
        end
    end
    
    % Save contact point (in Z sensor units) for this iteration
    local_CP = Zsnsr(minxcndx);
    
    %% Iterative Trimming if Needed
    if trimmingNeeded
        % Compute full indentation depth after contact point
        D_full = Zsnsr(minxcndx:extndx) - Zsnsr(minxcndx);
        max_depth = D_full(end);
        candidate_depths = MIN_DEPTH_FOR_GOOD_CLASSIFICATION:50:max_depth;
        best_candidate_index = [];
        best_candidate_depth = -inf;
        for depth = candidate_depths
            idx = find(D_full <= depth, 1, 'last');
            if isempty(idx)
                continue;
            end
            candidate_index = minxcndx - 1 + idx;
            candidate_ext = Zsnsr(1:candidate_index);
            candidate_defl = defl(1:candidate_index);
            candidatePred = predict_NN_GUI_classification(candidate_ext, candidate_defl, networkModelClassification);
            if candidatePred < thresholdClassification
                best_candidate_depth = depth;
                best_candidate_index = candidate_index;
            end
        end
        if ~isempty(best_candidate_index)
            trimmed_ext = Zsnsr(1:best_candidate_index);
            trimmed_defl = defl(1:best_candidate_index);
            local_AcceptClipped = true;
        else
            trimmed_ext = fit_ext;
            trimmed_defl = fit_defl;
            local_AcceptClipped = false;
        end
    end
    
    %% Calculate Elastic Modulus using the (possibly trimmed) curve
    new_extndx = length(trimmed_ext);
    def_val = trimmed_defl(minxcndx:new_extndx) - trimmed_defl(minxcndx);
    D_new = trimmed_ext(minxcndx:new_extndx) - trimmed_ext(minxcndx);
    D_new = D_new - def_val;
    F_new = def_val .* spring_constant;
    
    [E_app, regimeChange] = calc_E_app(D_new, F_new, R, th, b, 'pointwise', PLOT_OPT, HERTZIAN_FRONT_REMOVE, TIP_IS_SPHERICAL);
    E_app = E_app * 1e18 * 1e-9;
    E_app = E_app / 1000;
    E_local = E_app(end);
    
    % Calculate fitted Hertzian Modulus using the trimmed curve
    try
        E_app_Hertz = calc_E_app(D_new, F_new, R, th, b, 'Hertz', 0, HERTZIAN_FRONT_REMOVE, TIP_IS_SPHERICAL );
        E_app_Hertz = E_app_Hertz * 1e18 * 1e-9;
        E_app_Hertz = E_app_Hertz / 1000;
        Hertzian_local = E_app_Hertz;
    catch
        Hertzian_local = NaN;
    end
    
    % Save outputs from this iteration into temporary arrays
    out_E{k}              = E_local;
    out_CP{k}             = local_CP;
    out_Hertzian{k}       = Hertzian_local;
    out_Accept{k}         = local_Accept;
    out_AcceptClipped{k}  = local_AcceptClipped;
    out_F{k}              = F_new;
    out_D{k}              = D_new;
    out_Ext{k}            = trimmed_ext;
    out_ExtDefl{k}        = trimmed_defl;
    out_ExtDeflClipped{k} = trimmed_defl;
    out_PWE{k}            = real(E_app);
end

% After the parfor loop, reassemble the outputs into matrices and cell arrays.
E_Matrix               = zeros(nRows, nCols);
CP_Matrix              = zeros(nRows, nCols);
HertzianModulus_Matrix = zeros(nRows, nCols);
AcceptRejectMap        = false(nRows, nCols);
AcceptRejectMapClipped = false(nRows, nCols);

F_Matrix               = cell(nRows, nCols);
D_Matrix               = cell(nRows, nCols);
Ext_Matrix             = cell(nRows, nCols);
ExtDefl_Matrix         = cell(nRows, nCols);
ExtDefl_Matrix_Clipped = cell(nRows, nCols);
PWE_Matrix             = cell(nRows, nCols);

for k = 1:totalCurves
    i2 = out_i2{k};
    i  = out_i{k};
    E_Matrix(i2, i)               = out_E{k};
    CP_Matrix(i2, i)              = out_CP{k};
    HertzianModulus_Matrix(i2, i) = out_Hertzian{k};
    AcceptRejectMap(i2, i)        = out_Accept{k};
    AcceptRejectMapClipped(i2, i) = out_AcceptClipped{k};
    F_Matrix{i2, i}               = out_F{k};
    D_Matrix{i2, i}               = out_D{k};
    Ext_Matrix{i2, i}             = out_Ext{k};
    ExtDefl_Matrix{i2, i}         = out_ExtDefl{k};
    ExtDefl_Matrix_Clipped{i2, i} = out_ExtDeflClipped{k};
    PWE_Matrix{i2, i}             = out_PWE{k};
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
        'Ext_Matrix', 'ExtDefl_Matrix', 'ExtDefl_Matrix_Clipped', 'Height_Matrix', ...
        'spring_constant', 'v', 'PWE_Matrix', "HertzianModulus_Matrix", ...
        'AcceptRejectMap', 'AcceptRejectMapClipped', 'CONTACT_METHOD_OPT','PREDICT_QUALITY_OPT', ...
        'thresholdClassification','networkModelClassification', ...
        'networkModel', 'TIP_IS_SPHERICAL');
end

%% Re-enable Warnings
warning on;
