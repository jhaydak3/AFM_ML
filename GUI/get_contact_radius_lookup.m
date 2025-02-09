function aVal = get_contact_radius_lookup(h, R, b, th)
%GET_CONTACT_RADIUS_LOOKUP Returns the contact radius 'a' for depth h,
% using a precomputed lookup table for the given (R, b, th).
%
%   h  - indentation depth in nm (scalar or vector)
%   R  - tip radius (nm)
%   b  - blunt radius (nm)
%   th - half-opening angle (radians)

    % 1) Build a filename unique to the triple (R,b,th). For example:
    %    We'll do minimal rounding, but you may want a more sophisticated scheme
    %    to avoid floating-point issues.
    R_rounded = round(R,2);  % e.g. round to 2 decimals
    b_rounded = round(b,2);
    th_rounded= round(th,4); % angles might need more decimals
    lookupFolder = 'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v4_CNI_predict\helperFunctions\';
    fileName = sprintf('lookup_R%g_b%g_th%g.mat', R_rounded, b_rounded, th_rounded);
    fullPath = fullfile(lookupFolder, fileName);

    % 2) Check if the file exists
    if exist(fullPath, 'file')
        % --- Load existing .mat ---
        s = load(fullPath, 'Dgrid', 'agrid');
        Dgrid = s.Dgrid;    % Nx1 vector of precomputed depths
        agrid = s.agrid;    % Nx1 vector of contact radii
    else
        % --- Build the lookup table, then save it ---
        fprintf('Creating new lookup table for (R=%g, b=%g, th=%g)\n', ...
                 R_rounded, b_rounded, th_rounded);

        % Decide your maximum range. For example:
        Dmax = 5000;    % nm or bigger if you anticipate deeper indentation
        Dstep = 0.5;    % resolution
        Dgrid = (0:Dstep:Dmax)';

        % Preallocate
        agrid = nan(size(Dgrid));

        % For each depth Dgrid(i), call your normal 'fsolve' logic once
        for iD = 1:length(Dgrid)
            Dj = Dgrid(iD);
            if Dj <= b^2 / R
                % Spherical regime => "a" formula can be derived or is trivial:
                %   a ~ sqrt(2*R*Dj) is the sphere contact radius. But you can use
                %   an approximate or exact expression. Typically, for pure Hertz sphere:
                %   contact radius a_sphere = sqrt(R * Dj).
                %   But the code below might match your existing condition.
                agrid(iD) = sqrt(b^2);  % or sqrt(R * Dj), depends on your logic
            else
                % Blunted cone => solve once
                aInit = b; % initial guess
                [aSol, ~] = get_contact_radius_Jon_v2(b, Dj, R, th, aInit);
                agrid(iD) = aSol;
            end
        end

        % Save to .mat
        save(fullPath, 'Dgrid', 'agrid', '-v7.3');
    end

    % 3) Interpolate to get 'aVal' for the requested 'h'
    %    If you expect vector 'h', you can handle that too.
    %    We'll do a simple clamp or assume h is within [0, max(Dgrid)].
    h = min(max(h, Dgrid(1)), Dgrid(end));  % clamp
    aVal = interp1(Dgrid, agrid, h, 'linear');   % Nx1 or scalar
end
