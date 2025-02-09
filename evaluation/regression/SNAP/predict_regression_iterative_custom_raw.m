function predict_regression_iterative_custom_raw()
% PREDICT_REGRESSION_ITERATIVE_CUSTOM_RAW
%
% Iteratively estimate AFM contact points from RAW data (rawExt, rawDefl)
% using a pyramidal model:
%
%    F = (1/sqrt(2)) * [E/(1 - v^2)] * tan(halfAngle) * (indent^2).
%
% E is kept in Pa (SI units). Indentation is computed in meters from nm.
% We do up to 20 fitting iterations per curve, refining contact point z0
% and modulus E each time.
%
% It expects the .mat to contain (at least):
%   rawExt        {1 x M}, each cell is an Nx1 vector (nm)
%   rawDefl       {1 x M}, each cell is an Nx1 vector (nm)
%   spring_constant [1 x M], each in N/nm
%   v              [1 x M], Poisson's ratio (unitless)
%   th             [1 x M], half-angle in radians
%   minExtValues   [1 x M], for CP fraction
%   maxExtValues   [1 x M], for CP fraction
%
% Outputs saved in "predicted_cp_iterative_raw.mat":
%   YPred_new (M x 1) - predicted CP in [0,1]
%   z0_nm     (M x 1) - final CP in nm
%   E_Pa      (M x 1) - final modulus in Pa
%
% Author: J. Haydak & ChatGPT
% Date:   2025-02-05
% -------------------------------------------------------------------------
close all

%% -------------- Configuration -------------- %%
dataFile  = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\training\regression_processed_files\processed_features_for_regression_All.mat";
outFile   = "predicted_cp_iterative_raw.mat";

% Max number of iterations
maxIter         = 20;
offsetFraction  = 0.3;   % for deflection offset
deflThreshold   = 2;    % nm, for initial CP guess
deflFitRange    = [0, 15];  % nm
plotDebug       = false; % set true to see final fits

%% -------------- Load -------------- %%
if ~isfile(dataFile)
    error('File not found: %s', dataFile);
end
S = load(dataFile);

% We need rawExt, rawDefl, spring_constant, v, th, minExtValues, maxExtValues
requiredFields = {'rawExt','rawDefl','spring_constant','v','th',...
    'minExtValues','maxExtValues','Y', 'b','R'};
for f = 1:numel(requiredFields)
    if ~isfield(S, requiredFields{f})
        error('Missing field "%s" in mat file.', requiredFields{f});
    end
end

rawExt    = S.rawExt;          % cell array {1 x M}
rawDefl   = S.rawDefl;         % cell array {1 x M}
kAll      = S.spring_constant; % [1 x M], N/m
vAll      = S.v;              % [1 x M]
thAll     = S.th;             % [1 x M], in radians
minExtVal = S.minExtValues;   % [1 x M]
maxExtVal = S.maxExtValues;   % [1 x M]
Y = S.Y;
b = S.b;
R = S.R;

numCurves = numel(rawExt);
if numCurves ~= numel(rawDefl)
    error('rawExt and rawDefl dimension mismatch.');
end
fprintf('Loaded %d curves from raw data.\n', numCurves);

%% -------------- Allocate -------------- %%
YPred_new = nan(numCurves,1);  % final CP fraction in [0,1]
z0_nm     = nan(numCurves,1);  % final CP in nm
E_Pa      = nan(numCurves,1);  % final modulus in Pa

%% -------------- Main Loop -------------- %%
parfor i = 1:numCurves
    i
    ext_i  = rawExt{i}(:);    % nm
    defl_i = rawDefl{i}(:);   % nm
    k_i    = kAll(i);         % N/m
    nu_i   = vAll(i);         % dimensionless
    halfA  = thAll(i);        % radians
    b_i = b(i);
    R_i = R(i);

    [z0_final, E_final] = iterativeCPfit_singleCurve( ...
        ext_i, defl_i, k_i, nu_i, halfA, ...
        maxIter, offsetFraction, deflThreshold, deflFitRange, ...
        plotDebug, b_i, R_i);

    z0_nm(i) = z0_final;
    E_Pa(i)  = E_final;   % store final in Pa

    % Convert z0 -> fraction in [0,1]
    loVal = minExtVal(i);
    hiVal = maxExtVal(i);
    denom = hiVal - loVal;
    if denom > 0
        cpFrac = (z0_final - loVal) / denom;
        % clamp
        cpFrac = max(min(cpFrac,1),0);
        YPred_new(i) = cpFrac;
    else
        YPred_new(i) = 0;
    end
    % realCP = loVal + (hiVal - loVal) * Y(i);
    % figure
    % hold all
    % plot(ext_i, defl_i, '*')
    % xline(z0_nm(i))
    % xline(realCP,'r')
end


goodMask = ~isnan(YPred_new);
fprintf('Done. %d/%d curves have valid CP.\n', sum(goodMask), numCurves);

%% -------------- Save & Plot -------------- %%
save(outFile, 'YPred_new','z0_nm','E_Pa');
fprintf('Saved results to "%s".\n', outFile);

figure('Name','Iterative CP from raw','NumberTitle','off');
histogram(YPred_new(goodMask), 'Normalization','pdf','BinWidth',0.02);
xlabel('CP fraction'); ylabel('PDF');
title('Iterative CP from Raw Data');
grid on;
saveas(gcf, 'Histogram_CP_iterative_raw.png');

% ---- Compute mean absolute error in fractional space ----
% We assume "Y" is your true CP fraction in [0,1].
% Make sure "Y" is the same length as YPred_new.
diffFraction = abs( YPred_new(goodMask) - Y(goodMask) );
MAE_fraction = mean(diffFraction);  % mean absolute error (fraction)
STD_fraction = std(diffFraction);

% ---- Compute mean absolute error in nm ----
% The "real" CP in nm is: realCP(i) = loVal + (hiVal - loVal)*Y(i).
realCP_nm = nan(numCurves,1);
for i = 1:numCurves
    loVal = minExtVal(i);
    hiVal = maxExtVal(i);
    denom = hiVal - loVal;
    if denom > 0
        realCP_nm(i) = loVal + denom * Y(i);
    else
        realCP_nm(i) = loVal; % fallback if denom <= 0
    end
end

diffNm = abs( z0_nm(goodMask) - realCP_nm(goodMask) );


% -- After computing diffFraction and diffNm --

% Mean Absolute Error (already computed)
MAE_fraction = mean(diffFraction);
STD_fraction = std(diffFraction);

MAE_nm  = mean(diffNm);
STD_nm  = std(diffNm);

% Also compute Mean Squared Error
MSE_fraction = mean(diffFraction.^2);
MSE_nm       = mean(diffNm.^2);

fprintf('\n--- Contact Point Errors ---\n');
fprintf('Normalized MAE = %.4f (± %.4f)\n', MAE_fraction, STD_fraction);
fprintf('Absolute MAE   = %.4f nm (± %.4f nm)\n', MAE_nm,  STD_nm);

fprintf('Normalized MSE = %.4f\n', MSE_fraction);
fprintf('Absolute MSE   = %.4f nm^2\n', MSE_nm);

end

% =================================================================
%                 Local Functions
% =================================================================

function [z0_final, E_final] = iterativeCPfit_singleCurve( ...
    ext, defl, k, nu, halfAngle, ...
    maxIter, offsetFrac, deflThreshold, deflFitRange, plotDebug, b, R)
% ITERATIVECPFIT_SINGLECURVE
%
% For a single curve's extension/deflection (in nm), do up to maxIter
% iterative rounds:
%   1) Subtract deflection offset (first offsetFrac).
%   2) Initial guess z0 from defl>deflThreshold.
%   3) Fit (z0,E) in deflection range deflFitRange, comparing F_data vs F_model.
%   4) Repeat until z0 converges or hits maxIter.
%
% Return final z0 (nm) and E (Pa).

N = length(ext);
nOff = max(1, floor(offsetFrac*N));
deflOffset = defl - mean(defl(1:nOff));  % remove offset

% initial guess for z0
idxThresh = find(deflOffset > deflThreshold, 1, 'first');
if isempty(idxThresh)
    idxThresh = round(N/2);
end
z0_guess = ext(idxThresh);
d0_guess = deflOffset(idxThresh);

% initial guess for E in Pa
E_guess = 10000;  % e.g. 1 MPa as a rough guess (adjust if needed)

z0_final = z0_guess;

E_final  = E_guess;

for iter = 1:maxIter
    [z0_new, E_new] = fitOneRound( ...
        ext, deflOffset, k, nu, halfAngle, ...
        z0_final, E_final, deflFitRange, d0_guess, ...
        plotDebug, b, R);

    if abs(z0_new - z0_final) < 1e-3
        % converged if z0 changes < 1 nm
        z0_final = z0_new;
        E_final  = E_new;
        break;
    end

    z0_final = z0_new;
    E_final  = E_new;
end
end

function [z0_fit, E_fit] = fitOneRound( ...
    ext, deflOffset, k, nu, halfAngle, ...
    z0init, Einit, deflFitRange, d0, doPlot, b, R)
% FITONEROUND
%   Single pass of least-squares fit for p=[E,z0], using data where
%   deflectionOffset is within [deflFitRange(1), deflFitRange(2)] nm.
%
%   Indentation = (extension - z0) - deflOffset.
%   Force = k * deflOffset.
%
%   Model:
%   Fmodel = (1/sqrt(2)) * [ E/(1 - nu^2 ) ] * tan(halfAngle) * [indent^2].
%   where 'E' is in Pa, extension/deflection in nm, so final Fmodel
%   must match the same units as Fdata.

lowD = deflFitRange(1);
hiD  = deflFitRange(2);

% Restrict to points where offset deflection is in [lowD, hiD]
inRange = (deflOffset-d0 >= lowD & deflOffset-d0 <= hiD);
if sum(inRange) < 5
    % Not enough data for a stable fit
    z0_fit = z0init;
    E_fit  = Einit;
    return;
end

xData = ext(inRange);         % extension (nm)
dData = deflOffset(inRange)- d0;  % deflection offset (nm)
Fdata = k .* dData;           % force data

% Initial guesses for the solver
p0 = [Einit, z0init];  % p(1) = E in Pa, p(2) = z0 in nm

% Lower & upper bounds (example values, adjust to your scenario)
lb = [0,   min(ext)];
ub = [200000 , max(ext)];  %

% Make sure everything is double precision
xData = double(xData);
dData = double(dData);
p0    = double(p0);
lb    = double(lb);
ub    = double(ub);
Fdata = double(Fdata);

opts = optimoptions('lsqcurvefit',...
    'Display','off',...
    'MaxIterations',5000, ...
    'MaxFunctionEvaluations',5e4);

% We create an anonymous function that includes 'dData' inside,
% so the model can do: indentation = (x - z0) - d.
modelHandle = @(p, x) forceModelPyramid( p, x, dData, nu, halfAngle, b, R );

[p_best, ~, ~, exitflag] = lsqcurvefit( ...
    modelHandle, p0, xData, Fdata, lb, ub, opts);

if exitflag <= 0
    % Fit didn’t converge well; revert to initial
    z0_fit = z0init;
    E_fit  = Einit;
    return;
end

% Extract best-fit parameters
E_fit  = p_best(1);  % Pa
z0_fit = p_best(2);  % nm

if doPlot
    figure; hold on;
    scatter(xData, Fdata, 20,'b','filled'); % In same units as Fdata
    xxFine = linspace(min(xData), max(xData), 200)';

    % Evaluate the model on a finer grid, re-using the same 'dData'?
    % Actually, for a "pretty" plot, we'd also want dDataFine. But
    % we only have deflection data at discrete points. So for a quick
    % check, let's just do it at the actual xData points sorted:
    [xSorted, iSort] = sort(xData);
    Fsort = modelHandle(p_best, xSorted);

    plot(xSorted, Fsort, 'r-','LineWidth',2);
    xlabel('Extension (nm)');
    ylabel('Force (??? units)');  % depends on your k units
    title(sprintf('z0=%.1f nm, E=%.3g Pa', z0_fit, E_fit));
    grid on;
end
end


function F = forceModelPyramid(p, x, d, nu, halfAngle, b, R)
% FORCEMODELPYRAMID
%   p(1) = E (Pa)
%   p(2) = z0 (nm)
%
%   indentation (nm) = (x - z0) - d.
%   clamp negative indentations to zero.
%   Convert nm -> m for the (indent^2) term if your final force is in SI units.
%
%   Fmodel = (1/sqrt(2)) * [ E / (1 - nu^2 ) ] * tan(halfAngle) * (indent_m^2).

E_pa = p(1);
z0_nm= p(2);

indent_nm = (x - z0_nm) - d;  % <--- KEY FIX
indent_nm(indent_nm < 0) = 0; % clamp negative

% Convert indentation from nm -> m (if you want Force in Newtons)
% but if your 'Fdata' is actually in nN, then skip or adjust carefully!
% Here I'll assume:
%    k is (nN/nm),
%    deflOffset in nm => Fdata is in nN,
% so let's keep indentation in nm => Force in nN:
%    Then E must be "some" consistent unit basis.
% In pure SI, you'd do indent_m = indent_nm * 1e-9 and E in Pa => Force in N.
% Make sure you unify your units if you truly want SI.
% For demonstration, let's keep them as "nm" and E in "Pa" => mismatch.
%
% If you REALLY want SI, do:
%    indent_m = indent_nm * 1e-9;
%    factor = (1/sqrt(2)) * (E_pa / (1 - nu^2)) * tan(halfAngle);
%    F = factor * (indent_m.^2);  % in Newtons
%
% So let's do the real SI approach properly below:

indent_m = indent_nm * 1e-9;  % nm -> m

factor = (1/sqrt(2)) * (E_pa / (1 - nu^2)) * tan(halfAngle);
%factor = (2 / pi) * (E_pa / (1-nu^2)) * tan(halfAngle);
%F_SI = factor .* (indent_m.^2);  % in Newtons

% If your Fdata is in Newtons, this is correct.
% If your Fdata is in nN, multiply by 1e9 to match:
%F = F_SI * 1e9;  % => nN

F = zeros(size(indent_m));

% Implement blunted cone from Azeloglu & Costa (2011)
% note that here we use SI units rather than nanoSI units as in
% calculateModuli. Blah blah dimensional analysis blah blah

b_m = b / 1E9;
R_m = R / 1E9;
th = halfAngle;

% 1) Build a mask for spherical vs blunted cone
sphereMask = (indent_m <= b_m^2 / R_m);
bluntMask  = ~sphereMask;

% 2) Preallocate F
F = zeros(size(indent_m));

%% ----------------- Spherical Regime ----------------- %%
if any(sphereMask)
    Dj_sphere = indent_m(sphereMask);
    % Vector formula for F (spherical):
    %   F_sphere = E_pa * [ (8/3)* sqrt(D^3 * R_m ) ] 
    % (based on your snippet's comment).
    F(sphereMask) = E_pa * ( (8/3) .* sqrt(Dj_sphere.^3 .* R_m) );
end

%% ----------------- Blunted Cone Regime ----------------- %%
if any(bluntMask)
    Dj_blunt = indent_m(bluntMask);

    % get_contact_radius_lookup can be vectorized if you adjust it to accept
    % multiple indentations at once and return a vector 'aVec'.
    % We'll pass Dj_blunt (in nm) => so we convert from meters to nm first.
    Dj_blunt_nm = Dj_blunt * 1e9;    % m -> nm

    % Vector call:
    aVec_nm = get_contact_radius_lookup(Dj_blunt_nm, R_m*1e9, b_m*1e9, th);
    % Now 'aVec_nm' is also Nx1. Convert to meters:
    a_mVec = aVec_nm * 1e-9;

    tm1 = a_mVec .* Dj_blunt;
    tm2 = (a_mVec.^2) ./ (2 * tan(th));
    tm2 = tm2 .* ( (pi / 2) - asin( b_m ./ a_mVec ) );
    tm3 = (a_mVec.^3) ./ (3 * R_m);

    tm4 = b_m / (2 * tan(th));
    tm4 = tm4 + ( (a_mVec.^2 - b_m^2) ./ (3 * R_m) );
    tm4 = tm4 .* sqrt( a_mVec.^2 - b_m^2 );

    % F_blunt = E_pa * [4 * ( tm1 - tm2 - tm3 + tm4 )]
    F_blunt = E_pa * 4 .* ( tm1 - tm2 - tm3 + tm4 );
    F(bluntMask) = F_blunt;
end
% Now if my units are correct, F is in normal Si units, ie N. But we want
% nN!

F = F * 1E9;

end
