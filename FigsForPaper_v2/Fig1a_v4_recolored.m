%% Fig1a_v4_recolored_script.m
% Force curve in black, annotated CP in black, all offsets in black.
% Standalone script—no functions.

%% CLEAR AND USER PARAMETERS
clear; clc; close all;

% Figure parameters
figWidth        = 8;            % inches
figHeight       = figWidth*0.6; % inches
fontName        = 'Arial';
fontSize        = 16;
lwMain          = 1.5;          % main line width
cpLineWidth     = 3;            % annotated CP line width (top plot)
legendFontSize  = 12;

% Data & plotting parameters
matFilePath     = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\AFM_data\Podocytes\CSA_04.mat";
row             = 2;
col             = 1;
offsetValues    = [-200, -100, 100, 200];
skipDepth_nm    = 100;          % nm

%% SET DEFAULT STYLING
set(0, ...
    'DefaultFigureUnits',   'inches', ...
    'DefaultAxesFontName',  fontName, ...
    'DefaultAxesFontSize',  fontSize, ...
    'DefaultAxesLineWidth', lwMain, ...
    'DefaultLineLineWidth', lwMain);

addpath('C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\helperFunctions');

%% LOAD DATA
load(matFilePath, 'Ext_Matrix', 'ExtDefl_Matrix', 'CP_Matrix', ...
     'spring_constant', 'R', 'v', 'th', 'b');

extension_raw  = Ext_Matrix{row,col};
deflection_raw = ExtDefl_Matrix{row,col};
CP_raw         = CP_Matrix(row,col);

% Shift to zero
extension_actual  = extension_raw  - extension_raw(1);
deflection_actual = deflection_raw - deflection_raw(1);
CP_actual         = CP_raw         - extension_raw(1);

%% CREATE FIGURE
figure('Position',[1 1 figWidth figHeight],'Color','w');

%--- TOP: Extension vs Deflection
axTop = subplot(2,1,1);
hold(axTop,'on'); grid(axTop,'on');

% raw curve in black
plot(axTop, extension_actual, deflection_actual, 'k*-', ...
     'MarkerSize',2, 'DisplayName','Force Curve');

% spacer for legend
plot(axTop, NaN,NaN,'w','LineStyle','none','DisplayName',' ');

% annotated CP in black, thick
xline(axTop, CP_actual, 'k', 'LineWidth', cpLineWidth, ...
      'DisplayName','Annotated CP');

% offsets in black, various styles
styles = {'-','--',':','-.'};
for k = 1:numel(offsetValues)
    val = offsetValues(k);
    style = styles{mod(k-1,4)+1};
    xline(axTop, CP_actual+val, 'k', 'LineStyle', style, ...
          'LineWidth', lwMain, ...
          'DisplayName', sprintf('Offset %+d nm', val));
end

xlabel(axTop,'Extension (nm)');
ylabel(axTop,'Deflection (nm)');
title(axTop,'Deflection vs. Extension');
legend(axTop,'Location','best','FontSize',legendFontSize);

% optional axis limits
xlim(axTop,[-100 5500]);
ylim(axTop,[-5 60]);

%--- BOTTOM: Pointwise Modulus
axBot = subplot(2,1,2);
hold(axBot,'on'); grid(axBot,'on');
title(axBot,'Pointwise Modulus');

% offset=0 (annotated CP) in black
plotPWE(axBot, CP_actual,  0, extension_actual, deflection_actual, ...
        spring_constant, R, v, th, b, skipDepth_nm, ...
        'k','-', lwMain, 'Annotated CP');

% other offsets
for k = 1:numel(offsetValues)
    val = offsetValues(k);
    style = styles{mod(k-1,4)+1};
    plotPWE(axBot, CP_actual, val, extension_actual, deflection_actual, ...
            spring_constant, R, v, th, b, skipDepth_nm, ...
            'k', style, lwMain, sprintf('Offset %+d nm', val));
end

xlabel(axBot,'Indentation depth (nm)');
ylabel(axBot,'Modulus (kPa)');
legend(axBot,'Location','best','FontSize',legendFontSize);
xlim(axBot,[0 1000]);
ylim(axBot,[0 50]);

%% LOCAL HELPER
function plotPWE(ax, CP_act, offsetVal, ext, defl, k_spring, R, v, th, b, skipD, ...
                 color, style, lw, legendLabel)
    % Compute shifted CP
    CPs = CP_act + offsetVal;
    [~, idx] = min(abs(ext - CPs));
    z0 = ext(idx); h0 = defl(idx);
    d1 = ext(idx:end)  - z0;
    d2 = defl(idx:end) - h0;
    depth = d1 - d2;
    force = d2 * k_spring;

    % clean NaN/Inf
    valid = depth >= 0 & isfinite(depth) & isfinite(force);
    depth = depth(valid);
    force = force(valid);
    if isempty(depth), return; end

    % pointwise modulus
    Ept = calc_E_app(depth, force, R, th, b, 'pointwise', 0, 0);
    E_kPa = Ept*1e18*1e-9/1000 * 2*(1-v^2);

    % only depths >= skipD
    idx2 = find(depth >= skipD);
    if isempty(idx2), return; end
    d2 = depth(idx2);
    E2 = E_kPa(idx2);

    % plot
    plot(ax, d2, E2, 'Color', color, 'LineStyle', style, ...
         'LineWidth', lw, 'DisplayName', legendLabel);
end
