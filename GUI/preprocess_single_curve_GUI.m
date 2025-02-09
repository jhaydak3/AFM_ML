function X = preprocess_single_curve_GUI(xData, yData, n_points)
% preprocess_single_curve
% Preprocesses a single extension-deflection curve into a feature matrix
% for input into the trained neural network.

% Inputs:
%   xData - Raw extension data (nm)
%   yData - Raw deflection data (nm)
%   n_points - Number of points for interpolation
%
% Output:
%   X - Feature matrix (numFeatures x sequenceLength)

%% **1. Compute Min/Max and Normalize Data**
minDeflection = min(yData);
maxDeflection = max(yData);
minExtension = min(xData);
maxExtension = max(xData);

% Normalize deflection
if (maxDeflection - minDeflection) == 0
    normDefl = zeros(size(yData));
else
    normDefl = (yData - minDeflection) / (maxDeflection - minDeflection);
end

% Normalize extension
if (maxExtension - minExtension) == 0
    normExt = zeros(size(xData));
else
    normExt = (xData - minExtension) / (maxExtension - minExtension);
end

%% **2. Interpolate Data to n_points**
extensionInterp = linspace(0, 1, n_points);
deflectionInterp = interp1(normExt, normDefl, extensionInterp, 'linear', 'extrap');

%% **3. Feature Extraction**
smoothFactor1 = 0.006;
smoothFactor2 = 0.005;
intervalRov = 35;

% Smoothed deflection curve
smoothedDefl = smoothdata(deflectionInterp, 'movmean', 'SmoothingFactor', smoothFactor1);
smoothedDefl = normalizeZeroToOne(smoothedDefl);

% Compute first derivative with eighth-order accuracy
eighthOrderDeriv = computeFirstDerivativeEighthOrder(smoothedDefl);
smoothedEighthOrderDeriv = smoothdata(eighthOrderDeriv, 'movmean', 'SmoothingFactor', smoothFactor2);
normalizedEighthOrderDeriv = normalizeZeroToOne(smoothedEighthOrderDeriv);

% Compute ratio of variance (RoV)
rov = generateRoV(smoothedDefl, intervalRov);
normalizedRov = normalizeZeroToOne(rov);

% Compute local linear fit
[slope, Rsq] = localLinearFit(smoothedDefl, intervalRov);
normalizedSlope = normalizeZeroToOne(slope);
normalizedRsq = normalizeZeroToOne(Rsq);

%% **4. Assemble Features Matrix**
X = [
    deflectionInterp;
    smoothedDefl';
    normalizedEighthOrderDeriv';
    normalizedRov';
    normalizedSlope';
    normalizedRsq'
];

end

%% Supporting Functions
% Ensure the following functions are defined and accessible in your MATLAB path:
% - computeFirstDerivativeEighthOrder
% - generateRoV
% - calc_E_app

function normalized = normalizeZeroToOne(x)
% Make sure it's a vector.
x = x(:);

maxX = max(x);
minX = min(x);

normalized = (x - minX) ./ (maxX - minX);
end


function deriv = computeFirstDerivativeEighthOrder(defl)
% Compute the 8th-order symmetric derivative
eighthOrderDeriv = zeros(size(defl));
h = 1;

for j = 5:(length(defl) - 4)
    eighthOrderDeriv(j) = (1/280 * defl(j-4) - 4/105 * defl(j-3) + 1/5 * defl(j-2) ...
        - 4/5 * defl(j-1) + 4/5 * defl(j+1) - 1/5 * defl(j+2) ...
        + 4/105 * defl(j+3) - 1/280 * defl(j+4)) / h;
end

% Handle boundaries by setting derivatives to zero
eighthOrderDeriv(1:4) = 0;
eighthOrderDeriv(end-3:end) = 0;
deriv = eighthOrderDeriv;
end

function deriv = computeSecondDerivativeEighthOrder(defl)
% computeSecondDerivativeEighthOrder
% Computes the second derivative of a deflection curve with 8th-order accuracy
%
% Syntax:
%   deriv = computeSecondDerivativeEighthOrder(defl)
%
% Inputs:
%   defl - Vector containing deflection data points
%
% Outputs:
%   deriv - Vector containing the second derivative of deflection
%
% Description:
%   This function computes the second derivative of a deflection curve using
%   a symmetric finite difference scheme with 8th-order accuracy. A 9-point stencil
%   is employed to achieve the desired accuracy. Boundary points where the stencil
%   cannot be fully applied are set to zero to maintain array size consistency.
%
% Example:
%   deflection = sin(linspace(0, 2*pi, 1000));
%   secondDeriv = computeSecondDerivativeEighthOrder(deflection);

% Ensure input is a column vector for consistency
defl = defl(:);

% Initialize the derivative vector
deriv = zeros(size(defl));

% Define step size (assuming uniform spacing; set to 1 if unknown)
h = 1;

% Define the coefficients for the 9-point stencil second derivative (8th-order accurate)
coeffs = [-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560];

% Length of the deflection data
N = length(defl);

% Compute the second derivative using the 9-point stencil
% Loop from the 5th point to the (N-4)th point to avoid boundary issues
for j = 5:(N-4)
    deriv(j) = (coeffs(1)*defl(j-4) + coeffs(2)*defl(j-3) + coeffs(3)*defl(j-2) + ...
        coeffs(4)*defl(j-1) + coeffs(5)*defl(j) + coeffs(6)*defl(j+1) + ...
        coeffs(7)*defl(j+2) + coeffs(8)*defl(j+3) + coeffs(9)*defl(j+4)) / (h^2);
end

% Handle boundaries by setting the second derivative to zero
deriv(1:4) = 0;
deriv(end-3:end) = 0;
end


function [slope, R2] = localLinearFit(vector, intervalSize)
% localLinearFit Performs a local linear fit on a vector using a sliding window.
%
%   [slope, R2] = localLinearFit(vector, intervalSize)
%
%   Inputs:
%       vector      - A numerical vector (row or column) containing data points.
%       intervalSize- An odd integer specifying the number of points in each sliding window.
%
%   Outputs:
%       slope       - A vector of the same size as 'vector' containing the slope of the local linear fit at each point.
%       R2          - A vector of the same size as 'vector' containing the R² value of the local linear fit at each point.
%
%   Example:
%       x = 1:100;
%       y = 2*x + randn(1,100); % y = 2x with some noise
%       [slope, R2] = localLinearFit(y, 5);
%       plot(x, y, 'b', x, slope, 'r', x, R2, 'g');
%       legend('Data', 'Local Slope', 'Local R^2');

% Input Validation
if nargin < 2
    error('Function requires two inputs: vector and intervalSize.');
end

if ~isvector(vector)
    error('Input "vector" must be a numerical vector.');
end

if ~isnumeric(intervalSize) || ~isscalar(intervalSize) || intervalSize < 2
    error('Input "intervalSize" must be a positive integer greater than or equal to 2.');
end

intervalSize = round(intervalSize); % Ensure intervalSize is integer

% Ensure intervalSize is odd for centering
if mod(intervalSize, 2) == 0
    warning('intervalSize must be an odd integer. Incrementing intervalSize by 1.');
    intervalSize = intervalSize + 1;
end

halfWindow = floor(intervalSize / 2);

% Convert vector to column vector for consistency
vector = vector(:);
numPoints = length(vector);

% Preallocate output vectors
slope = NaN(numPoints, 1);
R2 = NaN(numPoints, 1);

% Perform local linear fit for each point
for i = 1:numPoints
    % Define window boundaries
    windowStart = max(1, i - halfWindow);
    windowEnd = min(numPoints, i + halfWindow);
    windowIndices = windowStart:windowEnd;
    windowSizeActual = length(windowIndices);

    % Ensure there are at least two points to perform linear fit
    if windowSizeActual < 2
        slope(i) = NaN;
        R2(i) = NaN;
        continue;
    end

    % Define x and y for linear fit
    x = windowIndices;
    y = vector(windowIndices);

    % Perform linear regression: y = a*x + b
    p = polyfit(x, y, 1);
    slope(i) = p(1); % Extract slope

    % Compute R^2
    y_fit = polyval(p, x)';
    SS_res = sum((y - y_fit).^2);
    SS_tot = sum((y - mean(y)).^2);
    if SS_tot == 0
        R2(i) = NaN; % Undefined R^2
    else
        R2(i) = 1 - (SS_res / SS_tot);
    end
end

% Convert outputs to the original shape (row or column)
if isrow(vector)
    slope = slope';
    R2 = R2';
end
end

function peakCount = movcountpeaks(data, window)
n = length(data);
peakCount = zeros(size(data));
halfWindow = floor(window / 2);
for i = 1:n
    startIdx = max(1, i - halfWindow);
    endIdx = min(n, i + halfWindow);
    windowData = data(startIdx:endIdx);
    [pks, ~] = findpeaks(windowData);
    peakCount(i) = length(pks);
end
end


function skew = movskew(data, window)
n = length(data);
skew = zeros(size(data));
halfWindow = floor(window / 2);
for i = 1:n
    startIdx = max(1, i - halfWindow);
    endIdx = min(n, i + halfWindow);
    windowData = data(startIdx:endIdx);
    skew(i) = skewness(windowData);
end

end

function kurt = movkurt(data, window)
n = length(data);
kurt = zeros(size(data));
halfWindow = floor(window / 2);
for i = 1:n
    startIdx = max(1, i - halfWindow);
    endIdx = min(n, i + halfWindow);
    windowData = data(startIdx:endIdx);
    kurt(i) = kurtosis(windowData);
end
end
