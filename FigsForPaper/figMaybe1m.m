clear
clc
close all

%% Load the data 
matFile = "C:\Users\MrBes\Documents\MATLAB\Jon_AFM_Code\version4\Training\Tubules\WT_Dish01_2021Nov02.mat";
load(matFile)

%% Calculate relative height from CP_Matrix
% Here we subtract CP_Matrix from its maximum value to get a relative scale.
relativeHeight = max(CP_Matrix, [], 'all') - CP_Matrix;

%% Identify “bad” indices based on indentInfoMap quality
% Note: indentInfoMap is a 32x32 string array.
badMask = (indentInfoMap == "Bad vibes") | (indentInfoMap == "Ambiguous CP");

%% Create a cleaned version by setting bad entries to NaN
cleaned = relativeHeight;
cleaned(badMask) = NaN;

%% Interpolate over the bad (NaN) values
% Create a grid of x (column) and y (row) coordinates.
[numRows, numCols] = size(cleaned);
[X, Y] = meshgrid(1:numCols, 1:numRows);

% Use the good (non-bad) points to set up an interpolant.
F = scatteredInterpolant(X(~badMask), Y(~badMask), cleaned(~badMask), 'linear', 'nearest');

% Replace the NaNs (bad points) with the interpolated values.
cleaned_interp = cleaned;
cleaned_interp(badMask) = F(X(badMask), Y(badMask));

%% Create the figure with a tiled layout (1 row, 2 columns)
figure;
tiledlayout(1,2, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- Left panel: Raw Height Data ---
nexttile;
imagesc(Height_Matrix);  % or use heatmap(Height_Matrix) if preferred
axis image off;
colorbar;
title('Raw Height Data');

% --- Right panel: Cleaned Relative Height Data ---
nexttile;
imagesc(cleaned_interp);  % or use heatmap(cleaned_interp) if preferred
axis image off;
colorbar;
title('Cleaned Relative Height Data');

%% Optionally, add an overall title or caption for the tiled layout
sgtitle('AFM Data: Raw vs. Cleaned Heatmaps');

%% 3D Surface with 2D Contour Projection for Height Data
% Define the view angles
az = 17.7;
el = 36.4;

%% 3D Surface with 2D Contour Projection (Prettier Version)

% Create a new figure with two subplots side-by-side
figure;
tiledlayout(1,2, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- Left Subplot: Raw Height Data ---
nexttile;
% Use surfc to create a 3D surface with an integrated contour projection
surfc(Height_Matrix, 'EdgeColor', 'none');
colormap(magma);         % Apply the copper colormap
shading faceted;           % Smooth color transitions
camlight('headlight');    % Add a headlight-style light
%lighting gouraud;         % Use Gouraud lighting for smooth illumination
colorbar;
xlabel('X (pixels)', 'FontSize', 12);
ylabel('Y (pixels)', 'FontSize', 12);
zlabel('Height (nm)', 'FontSize', 12);
title('Raw Height: Surface & Contour', 'FontSize', 14);
view(az, el);             % Set the view to the specified angles

% --- Right Subplot: Filtered (Interpolated) Height Data ---
nexttile;
surfc(cleaned_interp, 'EdgeColor', 'none');
colormap(magma);         % Apply the copper colormap
shading faceted;           % Smooth color transitions
camlight('headlight');    % Add a headlight-style light
%lighting gouraud;         % Use Gouraud lighting for smooth illumination
colorbar;
xlabel('X (pixels)', 'FontSize', 12);
ylabel('Y (pixels)', 'FontSize', 12);
zlabel('Height (nm)', 'FontSize', 12);
title('Filtered Height: Surface & Contour', 'FontSize', 14);
view(az, el);             % Use the provided view angles

sgtitle('3D Surface with 2D Contour Projection (Copper Colorscheme)', 'FontSize', 16);




%% Section: Calculate and Interpolate Hertzian Moduli

% Preallocate a matrix for the Hertzian moduli.
% We assume the modulus map has the same size as CP_Matrix (32x32).
E_app_map = zeros(size(CP_Matrix));

% Set the amount of depth (in nm) to exclude from the front of the data.
% This is the same value used by your calc_E_app function in 'Hertz' mode.
HertzFrontRemoveAmount = 50;  % example value in nm


% Loop over each pixel to compute the Hertzian modulus using calc_E_app.
% Here, each pixel's depth and force curves are taken from D_Matrix and F_Matrix.
for r = 1:size(E_app_map, 1)
    for c = 1:size(E_app_map, 2)
        % Get the depth (D) and force (F) curves for this pixel.
        % (Assumes each cell contains a column vector.)
        D_curve = D_Matrix{r, c};
        F_curve = F_Matrix{r, c};
        
        % Calculate the Hertzian modulus for this pixel.
        % The call below uses your custom calc_E_app, which applies your conversion logic internally.
        % Inputs:
        %   D_curve                - Depth vector (in nm)
        %   F_curve                - Force vector (in N)
        %   R                      - Tip radius (in nm)
        %   th                     - Half-opening angle (in radians)
        %   b                      - Blunt radius (in nm)
        %   'Hertz'                - Mode: return a single Hertzian modulus value
        %   0                      - No plotting during calculation
        %   HertzFrontRemoveAmount - Depth (in nm) to exclude from the front
        E_app_val = calc_E_app(D_curve, F_curve, R, th, b, 'Hertz', 0, HertzFrontRemoveAmount);
        
        % Immediately apply the unit conversion.
        % (Your conversion is implemented exactly as in your function.)
        E_app_map(r, c) = E_app_val;
    end
end

E_app_map = E_app_map .* 1e18 .* 1e-9 ./ 1000;

%% Identify and Interpolate Over Flagged Quality Values
% These quality strings indicate data that should be interpolated.
badLabels = ["Ambiguous CP", "Bad vibes", "Precontact skibidi", "Probe slip", "Too stiff"];

% Create a logical mask from indentInfoMap (assumed to be a 32x32 string array)
badMask = ismember(indentInfoMap, badLabels);

% Create a "cleaned" Hertzian modulus map by marking flagged pixels as NaN.
E_app_cleaned = E_app_map;
E_app_cleaned(badMask) = NaN;

%% Interpolate the Bad-Quality Points
% Create a grid of x (column) and y (row) coordinates.
[numRows, numCols] = size(E_app_cleaned);
[X, Y] = meshgrid(1:numCols, 1:numRows);

% Build an interpolant using only the "good" data.
F_interp = scatteredInterpolant(X(~badMask), Y(~badMask), E_app_cleaned(~badMask), 'natural', 'boundary');

% Replace the flagged (NaN) values with the interpolated values.
E_app_interp = E_app_cleaned;
E_app_interp(badMask) = F_interp(X(badMask), Y(badMask));

%% Plot the Results: Raw vs. Interpolated Hertzian Moduli
figure;
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Left Panel: Raw Hertzian Modulus Map (Units: GPa)
nexttile;
imagesc(E_app_map);
axis image off;
colorbar;
title('Raw Hertzian Modulus Map (GPa)');

% Right Panel: Interpolated Hertzian Modulus Map (Units: GPa)
nexttile;
imagesc(E_app_interp);
axis image off;
colorbar;
title('Interpolated Hertzian Modulus Map (GPa)');

sgtitle('Hertzian Moduli: Raw vs. Interpolated');

%%
% Create a new figure with two subplots side-by-side
figure;
tiledlayout(1,2, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- Left Subplot: Raw Height Data ---
nexttile;
% Use surfc to create a 3D surface with an integrated contour projection
surfc(E_app_map, 'EdgeColor', 'none');
colormap(jet);         % Apply the copper colormap
shading faceted;           % Smooth color transitions
camlight('headlight');    % Add a headlight-style light
%lighting gouraud;         % Use Gouraud lighting for smooth illumination
colorbar;
xlabel('X (pixels)', 'FontSize', 12);
ylabel('Y (pixels)', 'FontSize', 12);
zlabel('Height (nm)', 'FontSize', 12);
title('Raw Height: Surface & Contour', 'FontSize', 14);
view(az, el);             % Set the view to the specified angles

% --- Right Subplot: Filtered (Interpolated) Height Data ---
nexttile;
surfc(E_app_interp, 'EdgeColor', 'none');
colormap(jet);         % Apply the copper colormap
shading faceted;           % Smooth color transitions
camlight('headlight');    % Add a headlight-style light
%lighting gouraud;         % Use Gouraud lighting for smooth illumination
colorbar;
xlabel('X (pixels)', 'FontSize', 12);
ylabel('Y (pixels)', 'FontSize', 12);
zlabel('Height (nm)', 'FontSize', 12);
title('Filtered Height: Surface & Contour', 'FontSize', 14);
view(az, el);             % Use the provided view angles

sgtitle('3D Surface with 2D Contour Projection (Copper Colorscheme)', 'FontSize', 16);

%% Single-Point Elasticity Modulus at 500 nm: Raw and Interpolated

% Fixed target indentation depth (in nm)
target_depth = 500;  

% Preallocate matrix for the single-point modulus.
% (Assume the modulus map has the same dimensions as CP_Matrix.)
E_single_map = zeros(size(CP_Matrix));

% Loop over each pixel (assumes D_Matrix and F_Matrix are cell arrays)
for r = 1:size(CP_Matrix,1)
    for c = 1:size(CP_Matrix,2)
        % Retrieve the depth and force curves for this pixel.
        % (Each cell is assumed to be a column vector.)
        D_curve = D_Matrix{r,c};  % Depth vector (nm)
        F_curve = F_Matrix{r,c};  % Force vector (N)
        
        % Interpolate to obtain the force at 500 nm.
        % (If target_depth is outside the measured range, interp1 will extrapolate.)
        force_500 = interp1(D_curve, F_curve, target_depth, 'linear', 'extrap');
        
        % Compute the modulus at 500 nm using your provided function.
        E_val = calc_E_singlePoint(target_depth, force_500, R, th, b);
        
        % Convert to kPa using your conversion:
        % Multiply by 1e18, then by 1e-9, then divide by 1000.
        % (i.e. E_val * 1e18 * 1e-9 / 1000)
        E_single_map(r, c) = E_val * 1e18 * 1e-9 / 1000;
    end
end

%% Flag Bad-Quality Data and Interpolate

% Quality flags that indicate unreliable modulus values
badLabels = ["Ambiguous CP", "Bad vibes", "Precontact skibidi", "Probe slip", "Too stiff"];

% Create a logical mask of bad pixels using the quality map
badMask_single = ismember(indentInfoMap, badLabels);

% Create a "cleaned" modulus map by marking flagged pixels as NaN
E_single_cleaned = E_single_map;
E_single_cleaned(badMask_single) = NaN;

% Interpolate over the NaNs using scatteredInterpolant
[numRows, numCols] = size(E_single_cleaned);
[X, Y] = meshgrid(1:numCols, 1:numRows);

F_interp = scatteredInterpolant(X(~badMask_single), Y(~badMask_single), ...
                                E_single_cleaned(~badMask_single), 'linear', 'nearest');

% Replace the bad-quality (NaN) values with interpolated ones
E_single_interp = E_single_cleaned;
E_single_interp(badMask_single) = F_interp(X(badMask_single), Y(badMask_single));

%% Plot the Raw and Interpolated Modulus Maps

figure;
tiledlayout(1,2, 'TileSpacing','compact','Padding','compact');

% Raw Single-Point Modulus at 500 nm
nexttile;
imagesc(E_single_map);
axis image;
colormap(magma);  % Use the copper colorscheme
colorbar;
title('Single-Point Modulus at 500 nm (Raw) [kPa]');
xlabel('X (pixels)');
ylabel('Y (pixels)');

% Interpolated Single-Point Modulus at 500 nm
nexttile;
imagesc(E_single_interp);
axis image;
colormap(magma);
colorbar;
title('Single-Point Modulus at 500 nm (Interpolated) [kPa]');
xlabel('X (pixels)');
ylabel('Y (pixels)');

sgtitle('Elastic Modulus at 500 nm: Raw vs. Interpolated');
