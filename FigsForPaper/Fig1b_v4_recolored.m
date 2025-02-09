function Fig1b_v4_recolored()
    % FIGUREA_AND_POINTWISEONLY
    %
    % Creates a figure with TWO subplots:
    %   (1) TOP: Extension vs. Deflection
    %       - Raw data in blue
    %       - One red vertical line at the actual CP
    %       - Four black vertical lines at offsets = -200, -100, 100, 200 nm,
    %         each with a unique line style.
    %
    %   (2) BOTTOM: "Pointwise Modulus"
    %       - Plots the actual CP offset (0 nm) in solid red
    %       - Plots the four offsets in black with the same unique line styles
    %         as in the top subplot
    %       - Skips the first 100 nm of indentation
    %
    % REQUIREMENTS:
    %   - "calc_E_app.m" in your MATLAB path (supports 'pointwise').
    %   - CP_Matrix(row,col) is the *raw* extension value at CP (unshifted).
    %


    clear; clc; close all;

    %% USER-DEFINED PARAMETERS
    % --------------------------------------------------------------
    matFilePath = 'C:\Users\MrBes\Documents\MATLAB\Jon_AFM_Code\version4\Training_CNI_and_PKD\CSA_04.mat';
    row = 2;
    col = 1;

    % Offsets to evaluate (for vertical lines & black PWE curves):
    offsetValues = [-200, -100, 100, 200];

    % We'll also plot offset=0 (actual CP) in the bottom subplot (red).
    skipDepth_nm = 100;  % skip any indentation < 100 nm

    % --------------------------------------------------------------

    %% (1) Load data
    load(matFilePath, ...
         'Ext_Matrix', ...
         'ExtDefl_Matrix', ...
         'CP_Matrix', ...
         'spring_constant', ...
         'R', ...
         'v', ...
         'th', ...
         'b');

    if ~exist('Ext_Matrix','var') || ~exist('ExtDefl_Matrix','var') || ~exist('CP_Matrix','var')
        error('Missing required variables (Ext_Matrix, ExtDefl_Matrix, CP_Matrix) in the .mat file.');
    end

    extension_raw  = Ext_Matrix{row, col};     
    deflection_raw = ExtDefl_Matrix{row, col}; 
    CP_raw         = CP_Matrix(row, col);      

    if length(extension_raw) ~= length(deflection_raw)
        error('Mismatch in extension vs. deflection lengths.');
    end

    % Shift so extension/deflection start at zero
    extension_actual  = extension_raw  - extension_raw(1);
    deflection_actual = deflection_raw - deflection_raw(1);
    CP_actual         = CP_raw         - extension_raw(1);

    %% (2) Create figure
    fig = figure('Name','Figure A + PointwiseOnly','NumberTitle','off','Color','w');

    %------------------------------------------------------------
    % TOP SUBPLOT: Extension vs Deflection
    %------------------------------------------------------------
    axTop = subplot(2,1,1);
    hold(axTop,'on'); grid(axTop,'on');

    % Plot raw data in blue
    plot(axTop, extension_actual, deflection_actual, ...
         'k*-', 'LineWidth',1, 'MarkerSize',2, ...
         'DisplayName','Force Curve');


    plot(axTop, NaN, NaN, 'w', 'LineStyle','none', ...
     'DisplayName',' ');

    % Actual CP in a red vertical line
    xline(axTop, CP_actual, 'Color',[0 0 0],'LineWidth',5, ...
        'DisplayName','Actual CP');
    

    % 4 unique line styles (for the 4 offsets)
    offsetLineStyles = {'-','--',':','-.'};  

    % Add black lines at each offset
    for kk = 1:length(offsetValues)
        offsetVal = offsetValues(kk);

        styleIdx = mod(kk-1, numel(offsetLineStyles)) + 1;
        thisStyle = offsetLineStyles{styleIdx};

        xline(axTop, CP_actual + offsetVal, ...
            'LineStyle', thisStyle, 'Color',[0.2    0.2    0.2], 'LineWidth',1.5, ...
            'DisplayName', sprintf('Offset = %s%d nm', ...
                         plusMinusSign(offsetVal), abs(offsetVal)));
    end

    xlabel(axTop, 'Extension (nm)');
    ylabel(axTop, 'Deflection (nm)');
    title(axTop, 'Deflection vs. Extension');
    legend(axTop, 'Location','best');

    % Optional axis limits
    xlim(axTop, [-100 5500]);
    ylim(axTop, [-5 60]);
    xticks([0:500:5500])

    %------------------------------------------------------------
    % BOTTOM SUBPLOT: Pointwise Modulus (skip < 100 nm)
    %------------------------------------------------------------
    axBot = subplot(2,1,2);
    hold(axBot,'on'); grid(axBot,'on');
    title(axBot, 'Pointwise Modulus');

    % We'll plot offset=0 (actual CP) in red, then the 4 offsets in black
    % with the same line styles.

    % 1) Plot offset=0 (actual CP) in solid red
    doPlotPWE(axBot, ...
              CP_actual, 0, ...
              extension_actual, deflection_actual, ...
              spring_constant, R, v, th, b, ...
              'Color',[0 0 0 .7],'LineWidth',1.8, 'LineStyle','-', ...
              'DisplayName','Offset = 0 nm (Actual CP)', skipDepth_nm);


    % 2) Plot the four offsets in black, each with a unique style
    for kk = 1:length(offsetValues)
        offsetVal = offsetValues(kk);
        styleIdx = mod(kk-1, numel(offsetLineStyles)) + 1;
        thisStyle = offsetLineStyles{styleIdx};

        doPlotPWE(axBot, ...
                  CP_actual, offsetVal, ...
                  extension_actual, deflection_actual, ...
                  spring_constant, R, v, th, b, ...
                  'Color',[0.2    0.2    0.2],'LineWidth',1.5, 'LineStyle',thisStyle, ...
                  'DisplayName', sprintf('Offset = %s%d nm', ...
                               plusMinusSign(offsetVal), abs(offsetVal)), ...
                  skipDepth_nm);
    end

    xlabel(axBot, 'Indentation depth (nm)');
    ylabel(axBot, 'Modulus (kPa)');
    legend(axBot, 'Location','best');
    xlim([0 1000])
    ylim([0 50])
    xticks([0:100:1000])

end


% --------------------------------------------------------------------
% Helper function: sign as "+" or "-"
% --------------------------------------------------------------------
function pm = plusMinusSign(val)
    if val >= 0
        pm = '+';
    else
        pm = '';
    end
end

% --------------------------------------------------------------------
% Helper function to compute & plot PWE for a given offset
% --------------------------------------------------------------------
function doPlotPWE(axH, CP_actual, offsetVal, ...
                   extension_actual, deflection_actual, ...
                   k_spring, R, v, th, b, ...
                   varargin)
% doPlotPWE - plots the pointwise modulus vs indentation depth for 
%             CP offset = (CP_actual + offsetVal).
%
%  SKIPPING any depths < skipDepth_nm in indentation.
%
% Usage:
%   doPlotPWE(axH, CP_actual, offsetVal, extension_actual, deflection_actual, ...
%             k_spring, R, v, th, b, 'Color','r','LineStyle','-', skipDepth_nm)
%
% The final argument must be skipDepth_nm (numeric).
% The rest are name/value pairs for "plot(...)" styling.

    % Extract skipDepth_nm from the end
    skipDepth_nm = varargin{end};
    plotArgs = varargin(1:end-1);  % name-value pairs for plot

    CP_shifted = CP_actual + offsetVal;

    % Find index of CP
    [~, idxCP] = min(abs(extension_actual - CP_shifted));
    z0 = extension_actual(idxCP);
    h0 = deflection_actual(idxCP);

    depth1 = extension_actual(idxCP:end)  - z0;
    depth2 = deflection_actual(idxCP:end) - h0;
    depth  = depth1 - depth2;   % indentation depth
    force  = depth2 .* k_spring;

    % Clean up any invalid data
    validMask = ~isnan(depth) & ~isinf(depth) & ...
                ~isnan(force) & ~isinf(force);
    depth = depth(validMask);
    force = force(validMask);

    if isempty(depth)
        return; 
    end

    % Compute pointwise E
    [Etemp, ~] = calc_E_app(depth, force, R, th, b, 'pointwise', 0, 0);

    % Convert to kPa, factor in (1 - v^2)
    E_kPa = Etemp * 1e18 * 1e-9 / 1000;  
    E_kPa = E_kPa * 2 * (1 - v^2);

    % Skip data below skipDepth_nm
    if skipDepth_nm > 0
        mask = (depth >= skipDepth_nm);
        depth = depth(mask);
        E_kPa = E_kPa(mask);
        if isempty(depth), return; end
    end

    % Plot
    plot(axH, depth, E_kPa, plotArgs{:});
end
