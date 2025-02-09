function [E_app, regimeChange, varargout] = calc_E_app(D, F, R, th, b, mode, plotOpt, HertzFrontRemoveAmount)
% Calculates E_app using blunted cone model.
% Inputs:
%   D - Depth vector Nx1 in nm
%   F - Force vector Nx1 in N
%   R - Tip radius in nm
%   th - Half-opening angle of the tip in radians
%   b - Blunt radius in nm
%   mode - 'pointwise' or 'Hertz' for pointwise or single value of E returned
%   plotOpt - 1 or 0 to plot or not plot the linear plot for Hertz analysis
%   HertzFrontRemoveAmount - Amount of depth (in nm) to exclude from the front of the data in 'Hertz' mode
%
% Outputs:
%   E_app - Vector of depthwise E_app values if mode is 'pointwise' or single value if mode is 'Hertz'
%   regimeChange - Index where the regime changes from spherical to blunted cone
%   varargout{1} - r^2 value for the linear fit if 'Hertz' mode is used

regimeChange = 0;

if strcmp(mode, 'pointwise')
    E_app = zeros(length(F), 1);
    for j = 1:length(D)
        % The solution for the modulus uses a spherical geometry until
        % the contact radius surpasses the cylindrical radius
        % and a blunt-cone geometry once this point is reached:
        Dj = D(j);
        if Dj <= b^2 / R % Spherical
            E_app(j) = F(j) / (8 / 3 * sqrt(Dj^3 * R));
        else % Blunted cone
            % Solve a nonlinear 1-D equation for contact area
            a = get_contact_radius_lookup(Dj, R, b, th);
            tm1 = a * Dj;
            tm2 = a^2 / (2 * tan(th));
            tm2 = tm2 * (pi / 2 - asin(b / a));
            tm3 = a^3 / (3 * R);
            tm4 = b / (2 * tan(th));
            tm4 = tm4 + ((a^2 - b^2) / (3 * R));
            tm4 = tm4 * sqrt(a^2 - b^2);
            E_app(j) = F(j) / (4 * (tm1 - tm2 - tm3 + tm4));

            a_guess = a;
            if regimeChange == 0
                regimeChange = j;
            end
        end
    end

elseif strcmp(mode, 'Hertz')
    % Ensure F is a column vector
    if size(F, 2) > size(F, 1)
        F = F';
    end
    % x_fit = zeros(size(F));
    % for j = 1:length(D)
    %     % The solution for the modulus uses a spherical geometry until
    %     % the contact radius surpasses the cylindrical radius
    %     % and a blunt-cone geometry once this point is reached:
    %     Dj = D(j);
    %     if Dj <= b^2 / R % Spherical
    %         x_fit(j) = (8 / 3 * sqrt(Dj^3 * R));
    %     else % Blunted cone
    %         % Solve a nonlinear 1-D equation for contact area
    %         a = get_contact_radius_lookup(Dj, R, b, th);
    %         tm1 = a * Dj;
    %         tm2 = a^2 / (2 * tan(th));
    %         tm2 = tm2 * (pi / 2 - asin(b / a));
    %         tm3 = a^3 / (3 * R);
    %         tm4 = b / (2 * tan(th));
    %         tm4 = tm4 + ((a^2 - b^2) / (3 * R));
    %         tm4 = tm4 * sqrt(a^2 - b^2);
    %         x_fit(j) = 4 * (tm1 - tm2 - tm3 + tm4);
    %         if regimeChange == 0
    %             regimeChange = j;
    %         end
    %     end
    % end
    % Suppose D and F are depth/force arrays (Nx1).
    % We want to compute x_fit (Nx1) for the linearized Hertz approach.
    % We also track the index 'regimeChange' which is the first index
    % where the code transitions from spherical to blunted cone.

    % 1) Create mask for spherical vs. blunted cone
    sphericalMask = (D <= b^2 / R);
    bluntedMask   = ~sphericalMask;

    % 2) Allocate x_fit
    x_fit = zeros(size(D));

    % 3) Handle the spherical portion in vector form
    %    The formula is x_fit = (8/3) * sqrt(D^3 * R) for the spherical regime.
    x_fit(sphericalMask) = (8/3) .* sqrt( D(sphericalMask).^3 .* R );

    % 4) Handle the blunted portion in vector form
    D_blunt = D(bluntedMask);
    if ~isempty(D_blunt)
        % a_blunt is the contact radius for each D_blunt
        a_blunt = get_contact_radius_lookup(D_blunt, R, b, th);

        % Now compute the same geometry terms for the entire vector:
        tm1 = a_blunt .* D_blunt;
        tm2 = (a_blunt.^2) ./ (2 * tan(th));
        tm2 = tm2 .* ( (pi/2) - asin(b ./ a_blunt) );
        tm3 = (a_blunt.^3) ./ (3 * R);

        tm4 = b / (2 * tan(th));
        tm4 = tm4 + ((a_blunt.^2 - b^2) ./ (3 * R));
        tm4 = tm4 .* sqrt(a_blunt.^2 - b^2);

        % x_fit for the blunted region
        x_fit(bluntedMask) = 4 * (tm1 - tm2 - tm3 + tm4);
    end

    % 5) regimeChange is the *first* index where we switch from spherical to blunted
    %    i.e., the first true in 'bluntedMask'
    regimeChange = find(bluntedMask, 1, 'first');
    if isempty(regimeChange)
        regimeChange = 0;  % means it never entered the blunted region
    end


    % Exclude data where D < HertzFrontRemoveAmount
    validIdx = D >= HertzFrontRemoveAmount;

    % Check if there are enough data points after exclusion
    if sum(validIdx) < 2
        error('Not enough data points after excluding the initial indentation range. Increase the number of data points or reduce HertzFrontRemoveAmount.');
    end

    % Linear fit to the remaining data
    x_fit_valid = x_fit(validIdx);
    F_valid = F(validIdx);

    % Fit without intercept (y = m * x)
    E_app = (x_fit_valid' * x_fit_valid) \ (x_fit_valid' * F_valid);

    % Calculate R^2
    F_fit = E_app * x_fit_valid;
    SS_res = sum((F_valid - F_fit).^2);
    SS_tot = sum((F_valid - mean(F_valid)).^2);
    rsq = 1 - SS_res / SS_tot;
    varargout{1} = rsq;

    if plotOpt == 1
        figure;
        subplot(2, 1, 1);
        plot(D, F, '-*');
        title('Force Curve');
        xlabel('Depth (nm)');
        ylabel('Force (N)');
        set(gca, 'fontsize', 14);

        subplot(2, 1, 2);
        plot(x_fit, F, '*');
        title('Linearized Hertz');
        xlabel('x');
        ylabel('Force (N)');
        hold on;
        plot(x_fit_valid, F_fit, 'r-', 'LineWidth', 1.5);
        legend('Raw Data', 'Linear Fit', 'Location', 'best');
        s = sprintf('R^2 = %1.3f', rsq);
        text(0.1, 0.8, s, 'Units', 'normalized', 'fontsize', 16);
        set(gca, 'fontsize', 18);
        hold off;
    end
else
    error('Unknown mode.');
end
end
