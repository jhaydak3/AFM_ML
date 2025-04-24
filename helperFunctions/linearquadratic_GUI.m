function predictedCP = linearquadratic_GUI(xData, yData, dataFraction, lb, ub)
    % Ensure valid input
    if isempty(xData) || isempty(yData)
        warning('Empty xData or yData. Returning NaN.');
        predictedCP = NaN;
        return;
    end

    % Normalize deflection
    normDefl = (yData - min(yData)) / (max(yData) - min(yData));
    normExt = (xData - min(xData)) / (max(xData) - min(xData));
    
    xi = linspace(0, 1, 2000);
    interpDefl = interp1(normExt,normDefl,xi);

    


    normDefl = interpDefl;
    normExt = xi;

    % Filter data based on deflection threshold
    validIdx = normDefl <= dataFraction;
    xFit = double(normExt(validIdx));
    yFit = double(normDefl(validIdx));

    % Check if there's enough data for fitting
    if numel(xFit) < 5
        warning('Insufficient data for curve fitting. Returning NaN.');
        predictedCP = NaN;
        return;
    end

    % Update upper bound based on xFit range
    ub(1) = xFit(end);

    % Initial guess
    p0 = (lb + ub) / 2;

    % Fit the model using lsqcurvefit
    fitOptions = optimoptions('lsqcurvefit', ...
        'Display', 'off', ...
        'MaxFunctionEvaluations', 1e6, ...
        'MaxIterations', 1e6, ...
        'FunctionTolerance', 1e-10);

    % Perform fitting
    [pBest, ~, residual, exitflag] = lsqcurvefit(@piecewiseFun, double(p0), xFit, yFit, double(lb), double(ub), fitOptions);

    % Validate fit
    if exitflag <= 0
        warning('Fit did not converge. Returning NaN.');
        predictedCP = NaN;
        return;
    end

    % Extract predicted contact point
    predictedCP = pBest(1);

    predictedCP = predictedCP*(max(xData) - min(xData)) + min(xData);
    predictedCP = double(predictedCP);
end

function fvals = piecewiseFun(params, x)
% piecewiseFun:
%   params(1) = c (contact point)
%   params(2) = linSlope
%   params(3) = polyExp
%   params(4) = polyAmp
%   params(5) = polyAmp2
%   params(6) = linConstant
%
% For x <= c:
%   f(x) = linSlope * x + linConstant
%
% For x > c:
%   f(x) = linConstant + linSlope*c + polyAmp*(x-c)^polyExp +
%   polyAmp2*(x-c)^polyExp2

    cVal        = params(1);
    linSlope    = params(2);
    polyExp     = params(3);
    polyAmp     = params(4);
    polyAmp2    = params(5);
    linConstant = params(6);
    polyExp2    = params(7);

    fvals = zeros(size(x));

    idxLin  = (x <= cVal);
    idxPoly = (x >  cVal);

    % Linear portion
    fvals(idxLin) = linSlope .* x(idxLin) + linConstant;

    % Polynomial portion
    if any(idxPoly)
        xPoly  = x(idxPoly);
        fAtC   = linSlope*cVal + linConstant;  % continuity at x = c
        fvals(idxPoly) = fAtC + polyAmp.*(xPoly - cVal).^polyExp + ...
                                 polyAmp2.*(xPoly - cVal).^polyExp2;
    end
end
