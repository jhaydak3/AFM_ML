function predictedCP = SNAP_GUI(xData, yData, k, nu, th, ...
        maxIter, offsetFraction, deflThreshold, deflFitRange, plotDebug, b, R)
    % Ensure valid input
    if isempty(xData) || isempty(yData)
        warning('Empty xData or yData. Returning NaN.');
        predictedCP = NaN;
        return;
    end

    % Convert inputs to double precision
    xData = double(xData);
    yData = double(yData);
    k = double(k);
    nu = double(nu);
    th = double(th);
    b = double(b);
    R = double(R);

    % Remove deflection offset
    N = length(xData);
    nOff = max(1, floor(offsetFraction * N));
    deflOffset = yData - mean(yData(1:nOff));

    % Initial guess for z0
    idxThresh = find(deflOffset > deflThreshold, 1, 'first');
    if isempty(idxThresh)
        idxThresh = round(N / 2);
    end
    z0_guess = xData(idxThresh);
    d0_guess = deflOffset(idxThresh);

    % Initial guess for E in Pa
    E_guess = 10000;  % e.g., 1 MPa

    z0_final = z0_guess;
    E_final  = E_guess;

    % Iterative fitting process
    for iter = 1:maxIter
        [z0_new, E_new] = fitOneRound( ...
            xData, deflOffset, k, nu, th, ...
            z0_final, E_final, deflFitRange, d0_guess, ...
            plotDebug, b, R);

        % Convergence check
        if abs(z0_new - z0_final) < 1e-3
            z0_final = z0_new;
            E_final  = E_new;
            break;
        end

        z0_final = z0_new;
        E_final  = E_new;
    end

    predictedCP = z0_final;  % Return CP in nm
end


function [z0_fit, E_fit] = fitOneRound( ...
    xData, deflOffset, k, nu, th, ...
    z0init, Einit, deflFitRange, d0, plotDebug, b, R)

    lowD = deflFitRange(1);
    hiD  = deflFitRange(2);

    inRange = (deflOffset - d0 >= lowD & deflOffset - d0 <= hiD);
    if sum(inRange) < 5
        z0_fit = z0init;
        E_fit  = Einit;
        return;
    end

    xFit = xData(inRange);
    dFit = deflOffset(inRange) - d0;
    Ffit = k .* dFit;

    % Initial parameter guesses
    p0 = [Einit, z0init];
    lb = [0, min(xData)];
    ub = [200000, max(xData)];

    % Ensure all data is double
    xFit = double(xFit);
    dFit = double(dFit);
    p0   = double(p0);
    lb   = double(lb);
    ub   = double(ub);
    Ffit = double(Ffit);

    opts = optimoptions('lsqcurvefit', ...
        'Display', 'off', ...
        'MaxIterations', 5000, ...
        'MaxFunctionEvaluations', 5e4);

    modelHandle = @(p, x) forceModelPyramid(p, x, dFit, nu, th, b, R);

    [p_best, ~, ~, exitflag] = lsqcurvefit( ...
        modelHandle, p0, xFit, Ffit, lb, ub, opts);

    if exitflag <= 0
        z0_fit = z0init;
        E_fit  = Einit;
        return;
    end

    E_fit  = p_best(1);
    z0_fit = p_best(2);

    if plotDebug
        figure;
        hold on;
        scatter(xFit, Ffit, 20, 'b', 'filled');
        xSorted = sort(xFit);
        Fsort = modelHandle(p_best, xSorted);
        plot(xSorted, Fsort, 'r-', 'LineWidth', 2);
        xlabel('Extension (nm)');
        ylabel('Force (nN)');
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
