%% Modified Iterative Regression Script Using Available MAT-File Fields
% This script loads a MAT-file that contains Ext_Matrix, ExtDefl_Matrix,
% CP_Matrix, spring_constant, v, th, b, and R. It then converts the cell arrays
% into 1xM cell arrays, computes per‑curve min/max extension values, and calls an
% iterative regression method (pyramidal model) to estimate the contact point (CP)
% in nm. Finally, it reshapes the CP values into a matrix, inverts them, and displays
% the result.
%
% (Replace the dummy get_contact_radius_lookup function with your actual code.)
%
% Author: Your Name
% Date:   2025-02-06

%% Select the file to process:
fh = "D:\AFM_PKD\211104-Gusella-PKD\ITG_Dish_03_2021Nov04.mat";
fh = "D:\AFM_PKD\211102-Gusella-PKD\NotValidated\DKO_Dish01_2021Nov02.mat";

%% Load the file and repackage variables
S = load(fh);

% Verify required fields.
requiredFields = {'Ext_Matrix','ExtDefl_Matrix','CP_Matrix','spring_constant','v','th','b','R'};
parfor f = 1:numel(requiredFields)
    if ~isfield(S, requiredFields{f})
        error('Missing field "%s" in file "%s".', requiredFields{f}, fh);
    end
end

% Reshape the 32x32 cell arrays into 1xM cell arrays.
rawExt   = reshape(S.Ext_Matrix, 1, []);     % each cell: extension curve (nm)
rawDefl  = reshape(S.ExtDefl_Matrix, 1, []);   % each cell: deflection curve
CP_true  = reshape(S.CP_Matrix, 1, []);         % annotated CP (fraction) [optional]

numCurves = numel(rawExt);

% Compute min and max extension for each curve (to convert CP fraction to nm later).
minExtValues = nan(1, numCurves);
maxExtValues = nan(1, numCurves);
for i = 1:numCurves
    % Ensure the cell is nonempty
    if ~isempty(rawExt{i})
        minExtValues(i) = min(rawExt{i});
        maxExtValues(i) = max(rawExt{i});
    end
end

% Use the scalar parameters from the file.
k_all = S.spring_constant;  % spring constant (assumed same for all curves)
v_all = S.v;                % Poisson's ratio
th_all = S.th;              % half-angle (radians)
b_all = S.b;                % blunt radius
R_all = S.R;                % tip radius

% (If CP_true is not used further in the iterative fitting, that's fine.)

%% Call the iterative regression method over all curves.
[YPred_new, z0_nm, E_Pa] = iterativeCPfit_allCurves(rawExt, rawDefl, k_all, v_all, th_all, minExtValues, maxExtValues, CP_true, b_all, R_all);

% If the number of curves is a perfect square, reshape the CP (in nm) into a matrix.
gridSize = round(sqrt(numCurves));
if gridSize^2 == numCurves
    newH = reshape(z0_nm, [gridSize, gridSize]);
else
    newH = z0_nm;  % leave as a vector if not square
end

%% Display the result with proper µm scaling and symmetric axes
close all;
% Invert the CP values as in your original code.
newH_disp = max(newH(:)) - newH;

% Define scan parameters (assumed from metadata)
totalScan_um = 62;  % total scan size in micrometers (µm)
nPoints = size(newH, 1);  % assuming newH is 32x32

% Create symmetric axis vectors (0 in the middle)
x = linspace(0, totalScan_um, nPoints);
y = linspace(0, totalScan_um, nPoints);

% Plot the heatmap with x and y axes in µm
figure;
imagesc(x, y, newH_disp);
set(gca, 'YDir', 'normal');  % ensure y-axis increases upward
colormap(gray);
colorbar;
xlabel('X (µm)');
ylabel('Y (µm)');
title('Relative Height (nm)');

clim([0 7500])
%% --- Calculate Hertzian Modulus for Each Curve and Display Heatmap ---
% Define the amount of depth (in nm) to exclude from the front in 'Hertz' mode.
HertzFrontRemoveAmount = 100;  % adjust as needed

% Preallocate an array for the Hertzian modulus (in kPa) for each curve.
HertzianModulusActual = nan(1, numCurves);

for i = 1:numCurves
    % Extract the original extension and deflection curves for curve i.
    ext_i = rawExt{i};    % extension in nm
    defl_i = rawDefl{i};  % deflection in nm

    % Retrieve the predicted contact point (in nm) for this curve from iterative fitting.
    cp = z0_nm(i);

    % Use only the data beyond the contact point.
    [~,ndx] = min(abs(ext_i - cp));

    depth1_actual = ext_i(ndx:end) - cp;        % [N - idx_actual +1 x 1]
    depth2_actual = defl_i(ndx:end) - defl_i(ndx);      % [N - idx_actual +1 x 1]
    depth_actual = depth1_actual - depth2_actual;                % [N - idx_actual +1 x 1]
    force_actual = depth2_actual * k_all;      % [N - idx_actual +1 x 1]

    D_actual = depth_actual;



    % Call your calc_E_app function in 'Hertz' mode to obtain the predicted modulus.
    % (Ensure that calc_E_app accepts vector inputs for D and F.)
    try
        E_hertz_val_actual = calc_E_app(D_actual, force_actual, R_all, th_all, b_all, 'Hertz', 0, HertzFrontRemoveAmount);
        HertzianModulusActual(i) = E_hertz_val_actual * 1e18 * 1e-9 / 1000;  % resulting in kPa
        HertzianModulusActual(i) = HertzianModulusActual(i)  * 2 * (1 - v_all^2);
    catch
        HertzianModulusActual(i) = nan;
    end

   
end

% If the number of curves forms a perfect square, reshape the Hertzian modulus array into a matrix.
if gridSize^2 == numCurves
    HertzianModulusMatrix = reshape(HertzianModulusActual, [gridSize, gridSize]);
else
    HertzianModulusMatrix = HertzianModulusActual;  % otherwise, leave it as a vector
end

%% Display the Hertzian modulus as a heatmap.
figure;
imagesc(x,y,HertzianModulusMatrix);
colormap(gray);
colorbar;
title('Hertzian Modulus (kPa)');
xlabel('X (µm)');
ylabel('Y (µm)');
set(gca, 'YDir', 'normal');  % ensure y-axis increases upward
clim([0 15])


%% -------------------- Local Functions -------------------- %%
function [YPred_new, z0_nm, E_Pa] = iterativeCPfit_allCurves(rawExt, rawDefl, k, nu, th, minExtValues, maxExtValues, Y_true, b, R)
    % iterativeCPfit_allCurves applies the iterative CP fitting to all curves.
    %
    % Inputs:
    %   rawExt, rawDefl: 1xM cell arrays (each cell is an Nx1 vector in nm)
    %   k:              spring constant (scalar)
    %   nu:             Poisson's ratio (scalar)
    %   th:             half-angle in radians (scalar)
    %   minExtValues, maxExtValues: 1xM vectors with min/max extension (nm) per curve
    %   Y_true:         1xM vector with annotated CP fraction (if available)
    %   b, R:         blunt radius and tip radius (scalars, in nm)
    %
    % Outputs:
    %   YPred_new: 1xM vector of predicted CP fractions in [0,1]
    %   z0_nm:     1xM vector of final CP (nm)
    %   E_Pa:      1xM vector of estimated moduli (Pa)
    
    numCurves = numel(rawExt);
    YPred_new = nan(1, numCurves);
    z0_nm = nan(1, numCurves);
    E_Pa = nan(1, numCurves);
    
    % Configuration for iterative fitting.
    maxIter = 20;
    offsetFraction = 0.3;      % fraction of points for deflection offset
    deflThreshold = 2;         % nm, threshold for initial CP guess
    deflFitRange = [0, 15];      % nm, range of deflection for fitting
    plotDebug = false;
    
    parfor i = 1:numCurves
        ext_i = double(rawExt{i}(:));   % ensure column vector (nm)
        defl_i = double(rawDefl{i}(:)); % column vector (nm)
        
        % If the curve is too short, skip.
        if length(ext_i) < 10 || length(defl_i) < 10
            continue;
        end
        
        % Call the iterative CP fitting for a single curve.
        [z0_final, E_final] = iterativeCPfit_singleCurve(ext_i, defl_i, k, nu, th, maxIter, offsetFraction, deflThreshold, deflFitRange, plotDebug, b, R);
        z0_nm(i) = z0_final;
        E_Pa(i) = E_final;
        
        % Convert z0 (nm) to CP fraction using the per-curve min and max.
        loVal = minExtValues(i);
        hiVal = maxExtValues(i);
        if hiVal > loVal
            cpFrac = (z0_final - loVal) / (hiVal - loVal);
            cpFrac = max(min(cpFrac, 1), 0);
            YPred_new(i) = cpFrac;
        else
            YPred_new(i) = 0;
        end
    end
end

function [z0_final, E_final] = iterativeCPfit_singleCurve(ext, defl, k, nu, halfAngle, maxIter, offsetFrac, deflThreshold, deflFitRange, plotDebug, b, R)
    % iterativeCPfit_singleCurve performs up to maxIter iterative rounds on one curve.
    N = length(ext);
    nOff = max(1, floor(offsetFrac * N));
    deflOffset = defl - mean(defl(1:nOff));  % remove offset
    
    % Initial guess for CP: first index where deflOffset > threshold.
    idxThresh = find(deflOffset > deflThreshold, 1, 'first');
    if isempty(idxThresh)
        idxThresh = round(N/2);
    end
    z0_guess = ext(idxThresh);
    d0_guess = deflOffset(idxThresh);
    
    % Initial guess for modulus E (in Pa)
    E_guess = 1e6;  % adjust as needed
    
    z0_final = z0_guess;
    E_final = E_guess;
    
    for iter = 1:maxIter
        [z0_new, E_new] = fitOneRound(ext, deflOffset, k, nu, halfAngle, z0_final, E_final, deflFitRange, d0_guess, plotDebug, b, R);
        if abs(z0_new - z0_final) < 1e-3  % convergence criterion (1e-3 nm)
            z0_final = z0_new;
            E_final = E_new;
            break;
        end
        z0_final = z0_new;
        E_final = E_new;
    end
end

function [z0_fit, E_fit] = fitOneRound(ext, deflOffset, k, nu, halfAngle, z0init, Einit, deflFitRange, d0, doPlot, b, R)
    % fitOneRound performs a single least-squares fit for [E, z0] on a single curve.
    lowD = deflFitRange(1);
    hiD = deflFitRange(2);
    
    inRange = (deflOffset - d0 >= lowD & deflOffset - d0 <= hiD);
    if sum(inRange) < 5
        z0_fit = z0init;
        E_fit = Einit;
        return;
    end
    
    xData = ext(inRange);              % extension (nm)
    dData = deflOffset(inRange) - d0;    % deflection offset (nm)
    Fdata = k .* dData;                % force (nN, assuming k is in N/m and d in nm)
    
    % Initial guesses: p(1)=E in Pa, p(2)=z0 in nm.
    p0 = [Einit, z0init];
    lb = [0, min(ext)];
    ub = [2e6, max(ext)];  % adjust upper bound for E if needed
    
    opts = optimoptions('lsqcurvefit', 'Display', 'off', 'MaxIterations', 5000, 'MaxFunctionEvaluations', 5e4);
    modelHandle = @(p, x) forceModelPyramid(p, x, dData, nu, halfAngle, b, R);
    
    [p_best, ~, ~, exitflag] = lsqcurvefit(modelHandle, p0, double(xData), double(Fdata), double(lb), double(ub), opts);
    if exitflag <= 0
        z0_fit = z0init;
        E_fit = Einit;
        return;
    end
    E_fit = p_best(1);
    z0_fit = p_best(2);
    
    if doPlot
        figure; hold on;
        scatter(xData, Fdata, 20, 'b', 'filled');
        [xSorted, iSort] = sort(xData);
        Fsort = modelHandle(p_best, xSorted);
        plot(xSorted, Fsort, 'r-', 'LineWidth', 2);
        xlabel('Extension (nm)');
        ylabel('Force (nN)');
        title(sprintf('z0 = %.1f nm, E = %.3g Pa', z0_fit, E_fit));
        grid on;
    end
end

function F = forceModelPyramid(p, x, d, nu, halfAngle, b, R)
    % forceModelPyramid computes the model force using a pyramidal (blunted cone)
    % model.
    %
    %   p(1) = E (Pa)
    %   p(2) = z0 (nm)
    %
    % x and d are in nm.
    E_pa = p(1);
    z0_nm = p(2);
    
    % Compute indentation: (extension - z0) minus deflection offset.
    indent_nm = (x - z0_nm) - d;
    indent_nm(indent_nm < 0) = 0;
    indent_m = indent_nm * 1e-9;  % convert nm to m
    
    % Convert b and R to meters.
    b_m = b / 1e9;
    R_m = R / 1e9;
    th = halfAngle;
    
    F = zeros(size(indent_m));
    
    % Spherical regime: when indentation (m) <= b_m^2/R_m.
    sphereMask = (indent_m <= b_m^2 / R_m);
    bluntMask = ~sphereMask;
    
    if any(sphereMask)
        Dj_sphere = indent_m(sphereMask);
        F(sphereMask) = E_pa * ((8/3) .* sqrt(Dj_sphere.^3 .* R_m));
    end
    
    if any(bluntMask)
        Dj_blunt = indent_m(bluntMask);
        % Convert indentation from m to nm for lookup.
        Dj_blunt_nm = Dj_blunt * 1e9;
        aVec_nm = get_contact_radius_lookup(Dj_blunt_nm, R_m*1e9, b_m*1e9, th);
        a_mVec = aVec_nm * 1e-9;
        
        tm1 = a_mVec .* Dj_blunt;
        tm2 = (a_mVec.^2) ./ (2 * tan(th));
        tm2 = tm2 .* ((pi/2) - asin(b_m ./ a_mVec));
        tm3 = (a_mVec.^3) ./ (3 * R_m);
        tm4 = b_m / (2 * tan(th));
        tm4 = tm4 + ((a_mVec.^2 - b_m^2) ./ (3 * R_m));
        tm4 = tm4 .* sqrt(a_mVec.^2 - b_m^2);
        F_blunt = E_pa * 4 .* (tm1 - tm2 - tm3 + tm4);
        F(bluntMask) = F_blunt;
    end
    
    % Convert force from SI units (N) to nN.
    F = F * 1e9;
end

