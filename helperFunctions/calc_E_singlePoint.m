function E_val = calc_E_singlePoint(depth_nm, force_N, R, th, b)
    % Returns a single elasticity modulus for one (depth, force).
    % Spherical vs. blunted cone check:
    if depth_nm <= b^2 / R
        % Spherical formula
        E_val = force_N / ((8/3) * sqrt(depth_nm^3 * R));
    else
        a = get_contact_radius_lookup(depth_nm, R, b, th);
        tm1 = a * depth_nm;
        tm2 = (a^2/(2*tan(th))) * (pi/2 - asin(b/a));
        tm3 = a^3/(3*R);
        tm4 = b/(2*tan(th));
        tm4 = tm4 + ((a^2 - b^2)/(3*R)) * sqrt(a^2 - b^2);
        denom = 4*(tm1 - tm2 - tm3 + tm4);
        E_val = force_N / denom;
    end
end
