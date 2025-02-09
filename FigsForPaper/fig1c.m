fh = "D:\AFM_PKD\211102-Gusella-PKD\NotValidated\INT_Dish02_2021Nov02.mat";
fh = "D:\AFM_PKD\211102-Gusella-PKD\NotValidated\DKO_Dish01_2021Nov02.mat";
fh = "D:\AFM_PKD\211104-Gusella-PKD\DKO_Dish_03_2021Nov04.mat"
fh = ["D:\AFM_PKD\211104-Gusella-PKD\ITG_Dish_01_2021Nov04.mat"]
fh = "D:\AFM_PKD\211104-Gusella-PKD\ITG_Dish_03_2021Nov04.mat"
newH = computeContactPoints(fh)
%newH = computeContactPoints("D:\AFM_PKD\211104-Gusella-PKD\WT_Dish_01_2021Nov04.mat");
%%
close all
newH = max(newH) - newH;

imagesc(newH)
colormap(parula)


function newH = computeContactPoints(matFile)
% computeContactPoints Computes contact points (in nm) for each AFM indentation curve.
%
%   newH = computeContactPoints(matFile) loads the specified .mat file, which
%   must contain the cell arrays Ext_Matrix and ExtDefl_Matrix (each 32x32). Each cell
%   holds a vector representing one AFM indentation curve, where Ext_Matrix contains the 
%   extension data (in nm) and ExtDefl_Matrix the corresponding deflection data.
%
%   For each curve, the function:
%     1. Performs min–max normalization to [0,1] and interpolates the data to 2000 points.
%     2. Truncates the data so that only points with normalized deflection <= dataFraction
%        are used.
%     3. Fits a piecewise model (using lsqcurvefit) to estimate the normalized contact point.
%     4. Converts the normalized contact point to nm using the original extension range.
%
%   The output newH is a 32×32 matrix containing the estimated contact points (nm).
%
%   Example:
%       newH = computeContactPoints("D:\AFM_PKD\211104-Gusella-PKD\WT_Dish_01_2021Nov04.mat");
%
%   See also lsqcurvefit.

    %% Load the Data
    if ~isfile(matFile)
        error('File "%s" not found.', matFile);
    end
    data = load(matFile);
    
    if ~isfield(data, 'Ext_Matrix') || ~isfield(data, 'ExtDefl_Matrix')
        error('The file must contain Ext_Matrix and ExtDefl_Matrix.');
    end
    
    Ext_Matrix    = data.Ext_Matrix;    % Extension curves (nm)
    ExtDefl_Matrix = data.ExtDefl_Matrix; % Corresponding deflection curves
    [nRows, nCols] = size(Ext_Matrix);
    
    % Preallocate the output matrix (contact points in nm)
    newH = nan(nRows, nCols);
    
    %% Configuration for the piecewise fit
    dataFraction = 0.3; % Use only points where normalized deflection <= 0.3
    nInterp = 2000;     % Number of points for interpolation
    
    % Parameter bounds for the piecewise model:
    % params: [c, linSlope, polyExp, polyAmp, polyAmp2, linConstant, polyExp2]
    lb = [0, -0.5, 2, 0, 0, 0, 1];
    ub = [1,  0.5, 2, 500, 500, 0, 1];
    
    % Options for lsqcurvefit
    fitOptions = optimoptions('lsqcurvefit',...
        'Display','off',...
        'MaxFunctionEvaluations',1e6,...
        'MaxIterations',1e6,...
        'FunctionTolerance',1e-10);
    
    %% Process Each Curve
    parfor i = 1:nRows
        for j = 1:nCols
            % Retrieve the original curves (assumed to be column vectors)
            extCurve = Ext_Matrix{i,j};      % Extension (nm)
            deflCurve = ExtDefl_Matrix{i,j};   % Deflection
            
            % Skip if the curve is too short
            if length(extCurve) < 10 || length(deflCurve) < 10
                continue;
            end
            
            %% Normalize to [0,1]
            ext_min = min(extCurve);
            ext_max = max(extCurve);
            if ext_max == ext_min
                continue;
            end
            normExt = (extCurve - ext_min) / (ext_max - ext_min);
            
            defl_min = min(deflCurve);
            defl_max = max(deflCurve);
            if defl_max == defl_min
                continue;
            end
            normDefl = (deflCurve - defl_min) / (defl_max - defl_min);
            
            %% Interpolate to 2000 Points
            origInd = linspace(0, 1, length(normExt));
            xi = linspace(0, 1, nInterp);
            normExt_interp  = interp1(origInd, normExt, xi, 'linear', 'extrap');
            normDefl_interp = interp1(origInd, normDefl, xi, 'linear', 'extrap');
            
            %% Truncate Data Using dataFraction
            validIdx = normDefl_interp <= dataFraction;
            xData = normExt_interp(validIdx);  % normalized extension
            yData = normDefl_interp(validIdx); % normalized deflection
            
            if length(xData) < 5
                % Not enough data in the region of interest.
                continue;
            end
            
            % Adjust the upper bound for the contact point parameter (c)
            thisUb = ub;
            thisUb(1) = xData(end);
            
            % Set initial guess as the midpoint of lb and thisUb
            p0 = 0.5 * (lb + thisUb);
            
            %% Fit the Piecewise Model
            try
                [pBest, ~, ~, exitflag] = lsqcurvefit(@piecewiseFun, p0, xData, double(yData), lb, thisUb, fitOptions);
            catch ME
                warning('Fitting failed for curve (%d,%d): %s', i, j, ME.message);
                continue;
            end
            
            if exitflag <= 0
                % The fit did not converge.
                continue;
            end
            
            % pBest(1) is the normalized contact point (c)
            cp_norm = pBest(1);
            % Convert normalized contact point to nm using the original extension range:
            cp_nm = cp_norm * (ext_max - ext_min) + ext_min;
            
            newH(i,j) = cp_nm;
        end
    end
    
    % Display summary of results.
    numFitted = sum(~isnan(newH(:)));
    fprintf('Estimated contact points for %d out of %d curves.\n', numFitted, nRows * nCols);
end

%% ----------------- Local Function ----------------- %%
function fvals = piecewiseFun(params, x)
% piecewiseFun Piecewise function for AFM contact point estimation.
%
%   params(1) = c          (normalized contact point)
%   params(2) = linSlope
%   params(3) = polyExp
%   params(4) = polyAmp
%   params(5) = polyAmp2
%   params(6) = linConstant
%   params(7) = polyExp2
%
% For x <= c:
%   f(x) = linSlope * x + linConstant
%
% For x > c:
%   f(x) = (linSlope * c + linConstant) + polyAmp*(x-c)^polyExp + polyAmp2*(x-c)^polyExp2

    cVal        = params(1);
    linSlope    = params(2);
    polyExp     = params(3);
    polyAmp     = params(4);
    polyAmp2    = params(5);
    linConstant = params(6);
    polyExp2    = params(7);
    
    fvals = zeros(size(x));
    
    % Define regions
    idxLin  = (x <= cVal);
    idxPoly = (x > cVal);
    
    % Linear portion for x <= c
    fvals(idxLin) = linSlope .* x(idxLin) + linConstant;
    
    % Polynomial portion for x > c (ensuring continuity at x = c)
    if any(idxPoly)
        xPoly = x(idxPoly);
        fAtC  = linSlope * cVal + linConstant;
        fvals(idxPoly) = fAtC + polyAmp .* (xPoly - cVal).^polyExp + polyAmp2 .* (xPoly - cVal).^polyExp2;
    end
end
