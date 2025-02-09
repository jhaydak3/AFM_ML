function [a, exitflag] = get_contact_radius_Jon_v2(b, h, R, t, aInit)
% GET_CONTACT_RADIUS_JON_V2  Numerically solve for contact radius 'a'
%   [a, exitflag] = get_contact_radius_Jon_v2(b, h, R, t, aInit)
%   uses 'aInit' as the initial guess for the solver instead of always 'b'.

    % If not provided, default to b (optional fallback):
    if nargin < 5
        aInit = b;
    end

    h = double(h);

    if R*h <= b^2
        warning('The requested depth is within the defect zone!');
    end

    % "options" can remain the same
    options = optimset('display', 'off', 'TolFun', 1e-12);

    F = @(aVal) contact_expression(aVal, b, h, R, t);

    [a, fval, exitflag] = fsolve(F, aInit, options);
    a = real(a);

    % Optional check on final residual
    if abs(fval) > 1e-5
        disp(['Warning: final residual was not near zero in get_contact_radius_Jon_v2. ' ...
              'fval=' num2str(fval)]);
    end
end

function F = contact_expression(a, b, h, R, t)
% Same as your original:
F = (h + (a./R).*(sqrt(a.^2 - b.^2) - a) ...
         -(a./tan(t)).*(0.5*pi - asin(b./a)));
end
