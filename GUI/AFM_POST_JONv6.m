%% Script to Process HDF5 Files from Asylum's .ARDF Converter
% This script processes the h5 file produced by using Asylum's .ARDF converter.
% It reads the data, extracts force curves, calculates contact points, and estimates
% elastic modulus using different methods.
%


% Disable warnings temporarily
warning off;

%% File Loading and Initialization

% Load the HDF5 file and read the info
info = h5info(h5_file_loc);
info2 = h5info(h5_file_loc, '/ForceMap');

% Get the dataset labels for the force curves
mdata = info2.Groups.Datasets;

% Combine dataset names into a cell array of size (N+1) x 1
FCnames = {mdata.Name}';  % Force Curve names

% Convert the string array FCnames into numerical indices for row and column
FCnames_str = string(FCnames(1:end-1));  % Exclude the last dataset ('segment')
sndx = strfind(FCnames_str, ':');  % Find the ':' separator
sndx = [sndx{:}]';  % Convert to column vector

rndx = zeros(length(FCnames_str), 1);  % Initialize row indices
cndx = rndx;  % Initialize column indices

% Parse row and column indices from force curve names
for j = 1:length(FCnames_str)
    this_string = char(FCnames_str(j));
    rndx(j) = str2double(this_string(1:sndx(j)-1));  % Row index
    cndx(j) = str2double(this_string(sndx(j)+1:end));  % Column index
end

%% Read Spring Constant from Metadata

% Read the spring constant from metadata attributes
metadata = info.Attributes.Value;
metadata = strsplit(metadata, '\n');
metadata = string(metadata);
TF = contains(metadata, 'SpringConstant: 0.');
metadata = metadata(TF);
string_w_sc = char(metadata(1));
pndx = strfind(string_w_sc, '.');
sc = string_w_sc(pndx-1:end);
spring_constant = str2double(sc);  % Spring constant in N/m or nN/nM

%% Initialize Matrices for Storing Results

% Initialize matrices to store elastic modulus and contact points
nRows = max(rndx) + 1;
nCols = max(cndx) + 1;
E_Matrix = zeros(nRows, nCols);  % Elastic modulus
CP_Matrix = E_Matrix;  % Contact points

% Additional matrices based on model selection
if MODEL_QUADRATIC_FIT == 2
    E_Matrix_raw = E_Matrix;  % Raw modulus matrix for model 2
elseif MODEL_QUADRATIC_FIT == 3
    rsq_Matrix = E_Matrix;  % R-squared values for model 3
end

% If saving results, initialize cell arrays for storing data
if SAVE_OPT == 1
    F_Matrix = cell(nRows, nCols);
    D_Matrix = F_Matrix;
    RoV_Matrix = F_Matrix;  % Ratio of variance
    RoVZ_Matrix = F_Matrix;  % Ratio of variance Z sensor
    Ext_Matrix = F_Matrix;  % Extension data for fitting
    ExtDefl_Matrix = F_Matrix;  % Z sensor of extension data
    if MODEL_QUADRATIC_FIT == 0
        PWE_Matrix = F_Matrix;  % Pointwise modulus
    end
end

%% Process Force Curves

% Read the segment data indicating where curves switch from extension to retraction
segmentdata = h5read(h5_file_loc, ['/ForceMap/0/' FCnames{end}]);

% Loop over force curves
tic;  % Start timer
for i = 1:nCols
    % Initialize temporary storage for current column
    thiscol = zeros(nRows, 1);
    thiscol_CP = thiscol;  % Contact points

    if MODEL_QUADRATIC_FIT == 2
        thiscol_raw = thiscol;  % Raw data for model 2
    elseif MODEL_QUADRATIC_FIT == 3
        thiscol_rsq = thiscol;  % R-squared values for model 3
    end

    if SAVE_OPT == 1
        % Initialize cell arrays for saving results
        thiscol_F = cell(nRows, 1);
        thiscol_D = thiscol_F;
        thiscol_RoV = thiscol_F;
        thiscol_RoVZ = thiscol_F;
        thiscol_Ext = thiscol_F;
        thiscol_ExtDefl = thiscol_F;
        if MODEL_QUADRATIC_FIT == 0
            thiscol_PWE = thiscol_F;
        end
    end

    

    for i2 = 1:nRows
        close all;  % Close all 

        if i2 == 3 && i == 4
            4;
        end

        % Generate string identifier for current force curve
        this = string([num2str(i2-1) ':' num2str(i-1)]);

        % Check if the current force curve exists
        if any(strcmp(FCnames_str, this))
            % Read data for the current force curve
            data = h5read(h5_file_loc, ['/ForceMap/0/' num2str(i2-1) ':' num2str(i-1)]);
        else
            continue;  % Skip if force curve does not exist
        end

        % Extract data columns: Raw, Deflection (converted to nm), Z sensor (converted to nm)
        raw = data(:, 1);
        defl = data(:, 2) / 1e-9;  % Convert to nm
        Zsnsr = data(:, 3) / 1e-9;  % Convert to nm

        % Extract extension and retraction indices
        ndx_1 = i2;  % Row index
        ndx_2 = i;  % Column index
        extndx = segmentdata(1, ndx_2, ndx_1);  % End of extension
        retndx = segmentdata(2, ndx_2, ndx_1);  % End of retraction

        % Plot raw data if plotting option is enabled
        if PLOT_OPT == 1
            figure;
            plot(Zsnsr(1:extndx), defl(1:extndx));
            title('Raw Data: Extension vs Deflection');
            xlabel('Extension (nm)');
            ylabel('Deflection (nm)');
        end

        % Define the fitting region
        start_idx = max(1, extndx - NUM_PTS_CONSIDERED + 1);
        fit_ext = Zsnsr(start_idx:extndx);
        fit_defl = defl(start_idx:extndx);

        % Check for non-monotonic extension data
        if any(diff(fit_ext) < 0)
            warning('The deflection curve contains some of the retraction portion.');
        end

        %% Find the Point of Contact

        if CONTACT_METHOD_OPT == 1
            % Linear-Quadratic Piecewise Fit Method
            % Estimate baseline deflection and standard deviation
            if NUM_PTS_TO_AVERAGE >= length(fit_defl)
                baseline_defl = mean(fit_defl(1:length(fit_defl)/2));
                baseline_std = std(fit_defl(1:length(fit_defl)/2));
            else
                baseline_defl = mean(fit_defl(1:NUM_PTS_TO_AVERAGE));
                baseline_std = std(fit_defl(1:NUM_PTS_TO_AVERAGE));
            end

            if baseline_std > MAX_STD_RAISE_ERROR
                warning('The standard deviation of the baseline deflection data set is higher than the specified maximum.');
            end

            % Determine cutoff index where deflection exceeds threshold
            fit_defl_temp = fit_defl - baseline_defl;
            cutoff_ndx = find(fit_defl_temp > MAX_DEFL_FIT, 1, 'first');
            if isempty(cutoff_ndx)
                fprintf('Objective deflection not obtained for row %i col %i\n', ndx_1, ndx_2);
                cutoff_ndx = length(fit_ext);
            end

            % Trim fitting region to the cutoff
            fit_ext = fit_ext(1:cutoff_ndx);
            fit_defl = fit_defl(1:cutoff_ndx);

            % Perform piecewise fitting to find contact point
            xcs_fit = 1:length(fit_ext);
            errvec = zeros(length(xcs_fit), 1);
            fits = cell(length(xcs_fit), 1);

            for j = 1:length(xcs_fit)
                [errvec(j), ~, fits{j}] = lsqfitFC_Jon(fit_ext, fit_defl, xcs_fit(j));
            end

            % Find the index of the minimum error
            [minerr, minxcndx_fit] = min(errvec);
            minxcndx = minxcndx_fit + start_idx - 1;

            % Plot results if plotting option is enabled
            if PLOT_OPT == 1
                figure;
                plot(fit_ext, fit_defl, '-*');
                hold on;
                plot(fit_ext, fits{minxcndx_fit}, '-');
                plot(fit_ext(minxcndx_fit), fit_defl(minxcndx_fit), 'ko', 'MarkerSize', 10);
                xlabel('Z sensor / Extension (nm)');
                ylabel('Deflection (nm)');
                title('Deflection Curve');
                legend('Raw', 'Fit', 'Contact Point', 'Location', 'best');
            end

        elseif CONTACT_METHOD_OPT == 2
            % Ratio of Variance Method
            leftpts = zeros(ROV_INTERVAL_N, length(fit_defl));
            rightpts = leftpts;

            for j = 1:ROV_INTERVAL_N
                leftpts(j, :) = circshift(fit_defl, j);
                rightpts(j, :) = circshift(fit_defl, -j);
            end

            leftvar = var(leftpts(:, ROV_INTERVAL_N+1:end-ROV_INTERVAL_N));
            rightvar = var(rightpts(:, ROV_INTERVAL_N+1:end-ROV_INTERVAL_N));
            ROV = rightvar ./ leftvar;

            [maxROV, minxcndx_fit] = max(ROV);
            minxcndx_fit = minxcndx_fit + ROV_INTERVAL_N;  % Account for offset
            minxcndx = minxcndx_fit + max(1, extndx - NUM_PTS_CONSIDERED);

            % Plot ROV if plotting option is enabled
            extplot = fit_ext(ROV_INTERVAL_N+1:end-ROV_INTERVAL_N);
            if length(extplot) < 2
                continue;
            end
            if PLOT_OPT == 1
                figure;
                subplot(2, 1, 1);
                plot(fit_ext, fit_defl, '-*');
                hold on;
                plot(fit_ext(minxcndx_fit), fit_defl(minxcndx_fit), 'ko', 'MarkerSize', 10);
                xlabel('Z sensor / Extension (nm)');
                ylabel('Deflection (nm)');
                legend('Raw data', 'Contact Point', 'Location', 'best');
                set(gca, 'FontSize', FontSize);

                subplot(2, 1, 2);
                plot(extplot, ROV);
                xlabel('Z sensor / Extension (nm)');
                ylabel('ROV');
                set(gca, 'FontSize', FontSize);
                title('Ratio of Variance');
            end

        elseif CONTACT_METHOD_OPT == 3
            %% ---------------------------------------------------------
            %  Method 3: Piecewise Fit with Normalization & Interpolation
            % ----------------------------------------------------------
            %
            % Steps:
            %   1) Normalize fit_ext and fit_defl to [0, 1].
            %   2) Interpolate onto 2000 points.
            %   3) Restrict domain to x ∈ [0, a], where deflection ≥ 0.1.
            %   4) Fit a piecewise model (linear for x ≤ c, polynomial for x > c).
            %   5) Convert the fitted contact c back to the original scale.
            %

            % -- 1) Normalize extension and deflection --
            extMin = min(fit_ext);
            extMax = max(fit_ext);
            defMin = min(fit_defl);
            defMax = max(fit_defl);

            % Guard against degenerate range
            if (extMax - extMin) < eps || (defMax - defMin) < eps
                fprintf('Near-constant curve at row %i col %i. Skipping contact fit.\n', ndx_1, ndx_2);
                minxcndx = start_idx;  % fallback
            else
                normExt  = (fit_ext  - extMin) / (extMax - extMin);
                normDefl = (fit_defl - defMin) / (defMax - defMin);

                % -- 2) Interpolate onto 2000 points from x=0..1
                xi = linspace(0, 1, 2000);
                deflInterp = interp1(normExt, normDefl, xi, 'linear', 'extrap');

                % -- 3) Restrict domain to portion up to deflection >= 0.1
                idxCut = find(deflInterp >= 0.1, 1, 'first');
                if isempty(idxCut)
                    % If we never reach 0.1, just use entire range
                    xFit = xi;
                    yFit = deflInterp;
                else
                    % Use from 0 up to that cut
                    xFit = xi(1:idxCut);
                    yFit = double(deflInterp(1:idxCut));
                end

                % If there's insufficient data left, revert to fallback
                if length(xFit) < 5
                    fprintf('Insufficient points for piecewise fit at row %i col %i.\n', ndx_1, ndx_2);
                    minxcndx = start_idx;  % fallback
                else
                    % -- 4) Fit the piecewise model on (xFit, yFit) --
                    % Parameter vector: [c, slopeLin, polyExp, polyAmp, polyAmp2]
                    lb = [ 0,      -0.5,     1,          100,        0   ];
                    ub = [ 1,       0.5,     4,          500,       50  ];
                    p0 = mean([lb; ub], 1);  % midpoint guess

                    lsqOpts = optimoptions('lsqcurvefit', ...
                        'Display','off', ...
                        'MaxIterations',2e3, ...
                        'MaxFunctionEvaluations',1e5);

                    [pBest, ~, ~, exitflag] = lsqcurvefit(@piecewiseFun_v3, p0, xFit, yFit, lb, ub, lsqOpts);

                    if exitflag <= 0
                        fprintf('Piecewise fit did NOT converge at row %i col %i.\n', ndx_1, ndx_2);
                        minxcndx = start_idx;  % fallback
                    else
                        % pBest(1) is the contact in normalized domain [0..1]
                        cNorm = pBest(1);

                        % -- 5) Convert cNorm -> original extension scale
                        cVal = cNorm*(extMax - extMin) + extMin;

                        % Find closest index in the un-interpolated fit_ext
                        [~, minxcndx_fit] = min(abs(fit_ext - cVal));
                        minxcndx = minxcndx_fit + start_idx - 1;

                        % (Optional) Plot
                        if PLOT_OPT == 1
                            figure('Name','Method 3: Piecewise Fit','NumberTitle','off');
                            hold on;
                            plot(fit_ext, fit_defl, 'b-*','DisplayName','Raw Data');

                            % Evaluate piecewise model on a fine grid 0..1
                            xFine = linspace(0,1,400);
                            yFine = piecewiseFun_v3(pBest, xFine);

                            % Map back to original scale for plotting
                            extPlot = xFine*(extMax - extMin) + extMin;
                            deflPlot = yFine*(defMax - defMin) + defMin;
                            plot(extPlot, deflPlot, 'r-','LineWidth',1.5,'DisplayName','Piecewise Fit');

                            % Mark contact
                            plot(fit_ext(minxcndx_fit), fit_defl(minxcndx_fit), ...
                                'ko','MarkerSize',8,'DisplayName','Contact Point');

                            xlabel('Z sensor / Extension (nm)');
                            ylabel('Deflection (nm)');
                            title(sprintf('Piecewise Fit (row=%i, col=%i)', ndx_1, ndx_2));
                            legend('Location','best');
                            grid on;
                        end
                    end
                end
            end

        end




        % Store contact point in physical units (Z sensor value)
        thiscol_CP(i2) = Zsnsr(minxcndx);

        %% Calculate Elastic Modulus

        % Calculate the indentation depth and deflection
        def = defl(minxcndx:extndx) - defl(minxcndx);  % h - h0
        D = Zsnsr(minxcndx:extndx) - Zsnsr(minxcndx);  % z - z0
        D = D - def;  % D = (z - z0) - (h - h0)
        F = def .* spring_constant;  % Force vector using Hooke's law, units nM * nN / nM = nN

        % Calculate the apparent elastic modulus using pointwise calculation
        [E_app, regimeChange] = calc_E_app(D, F, R, th, b, 'pointwise', PLOT_OPT);

        % Convert elastic modulus to kPa
        E_app = E_app * 1e18 * 1e-9;  % Convert to N/m^2 (Pa) (from nN / nm^2)
        E_app = E_app / 1000;  % Convert to kPa
        E = E_app .* 2 .* (1 - v^2);  % Correct for Poisson's ratio

        % Store results for the current curve
        thiscol(i2) = E(end);

        if MODEL_QUADRATIC_FIT == 2
            thiscol_raw(i2) = E_raw(end);  % Store raw modulus for model 2
        elseif MODEL_QUADRATIC_FIT == 3
            thiscol_rsq(i2) = rsq;  % Store R-squared for model 3
        end

        % Save raw data if save option is enabled
        if SAVE_OPT == 1
            thiscol_F{i2} = F;
            thiscol_D{i2} = D;
            thiscol_Ext{i2} = Zsnsr(1:extndx);
            thiscol_ExtDefl{i2} = defl(1:extndx);
            if MODEL_QUADRATIC_FIT == 0
                thiscol_PWE{i2} = real(E);
            end
            if CONTACT_METHOD_OPT == 2
                thiscol_RoV{i2} = ROV;
                thiscol_RoVZ{i2} = extplot;
            end
        end
    end

    % Update matrices with current column results
    E_Matrix(:, i) = thiscol;
    CP_Matrix(:, i) = thiscol_CP;
    if MODEL_QUADRATIC_FIT == 2
        E_Matrix_raw(:, i) = thiscol_raw;
    elseif MODEL_QUADRATIC_FIT == 3
        rsq_Matrix(:, i) = thiscol_rsq;
    end

    % Save matrices if save option is enabled
    if SAVE_OPT == 1
        F_Matrix(:, i) = thiscol_F;
        D_Matrix(:, i) = thiscol_D;
        RoV_Matrix(:, i) = thiscol_RoV;
        RoVZ_Matrix(:, i) = thiscol_RoVZ;
        Ext_Matrix(:, i) = thiscol_Ext;
        ExtDefl_Matrix(:, i) = thiscol_ExtDefl;
        if MODEL_QUADRATIC_FIT == 0
            PWE_Matrix(:, i) = thiscol_PWE;
        end
    end

    % Display progress
    fprintf('%.2f percent done processing.\n', i/nCols * 100);
end

toc;  % End timer

%% Plotting Results

% Remove any imaginary parts due to floating point errors
E_Matrix = real(E_Matrix);

% Plot elastic modulus maps
if MODEL_QUADRATIC_FIT == 0 || MODEL_QUADRATIC_FIT == 1
    figure;
    imagesc(E_Matrix);
    title('Elastic Modulus');
    c = colorbar; c.Label.String = 'kPa';
    set(gca, 'FontSize', FontSize, 'YDir', 'normal');
elseif MODEL_QUADRATIC_FIT == 2
    figure;
    subplot(1, 2, 1);
    imagesc(E_Matrix);
    title('Elastic Modulus (fit)');
    c = colorbar; c.Label.String = 'kPa';

    set(gca, 'FontSize', FontSize, 'YDir', 'normal');

    subplot(1, 2, 2);
    imagesc(E_Matrix_raw);
    title('Elastic Modulus (raw)');
    c = colorbar; c.Label.String = 'kPa';

    set(gca, 'FontSize', FontSize, 'YDir', 'normal');
elseif MODEL_QUADRATIC_FIT == 3
    figure;
    imagesc(E_Matrix);
    title('Elastic Modulus');
    c = colorbar; c.Label.String = 'kPa';

    set(gca, 'FontSize', FontSize, 'YDir', 'normal');

    figure;
    imagesc(rsq_Matrix);
    title('R^2');
    colorbar;
    set(gca, 'FontSize', FontSize, 'YDir', 'normal');
end

% Plot relative height map based on contact points
Height_Matrix = max(CP_Matrix(:)) - CP_Matrix;
figure;
imagesc(Height_Matrix);
title('Relative Height (CP estimate)');
c = colorbar; c.Label.String = 'nm';
set(gca, 'FontSize', FontSize, 'YDir', 'normal');

%% Save Results

if SAVE_OPT == 1
    save(SAVE_NAME, 'E_Matrix', 'F_Matrix', 'D_Matrix', 'CP_Matrix', ...
        'h5_file_loc', 'CONTACT_METHOD_OPT', 'R', 'th', 'b', ...
        'RoV_Matrix', 'RoVZ_Matrix', 'Ext_Matrix', 'ExtDefl_Matrix', 'Height_Matrix', ...
        'spring_constant', 'v');
    if MODEL_QUADRATIC_FIT == 0
        save(SAVE_NAME, 'PWE_Matrix', '-append');
    elseif MODEL_QUADRATIC_FIT == 3
        rsq_Matrix = real(rsq_Matrix);
        save(SAVE_NAME, 'rsq_Matrix', '-append');
    end
end

%% Re-enable Warnings
warning on;
function fvals = piecewiseFun_v3(params, x)
% piecewiseFun_v3:
%   params(1) = c     : contact point in [0..1]
%   params(2) = slope : linear slope for x <= c
%   params(3) = pExp  : polynomial exponent for x > c
%   params(4) = pAmp  : polynomial amplitude
%   params(5) = pAmp2 : linear amplitude in the polynomial region
%
% For x <= c:
%       f(x) = slope * x
% For x > c:
%       f(x) = slope*c + pAmp*(x - c)^pExp + pAmp2*(x - c)

cVal   = params(1);
slope  = params(2);
pExp   = params(3);
pAmp   = params(4);
pAmp2  = params(5);

fvals  = zeros(size(x));
idxLin  = (x <= cVal);
idxPoly = (x >  cVal);

% -- Linear portion
fvals(idxLin) = slope * x(idxLin);

% -- Polynomial portion
if any(idxPoly)
    xPoly = x(idxPoly);
    fAtC  = slope*cVal;  % ensure continuity at x = cVal
    fvals(idxPoly) = fAtC + pAmp.*(xPoly - cVal).^pExp + pAmp2.*(xPoly - cVal);
end
end
