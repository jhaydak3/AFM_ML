function out = generateRoV(deflection, filterSize)
    % Length of the input vector
    n = length(deflection);
    
    % Initialize the output vector with zeros
    out = zeros(size(deflection));
    
    % Iterate through each element of the vector where calculation is possible
    for i = (filterSize + 1):(n - filterSize)
        % Points behind the current index
        behind = deflection(i-filterSize:i-1);
        % Points ahead of the current index
        ahead = deflection(i+1:i+filterSize);
        
        % Calculate variances
        varBehind = var(behind);
        varAhead = var(ahead);
        
        % Calculate the ratio of variance ahead to variance behind
        if varBehind == 0
            % Avoid division by zero; handle according to your requirement
            out(i) = 0;  % Could also be set to Inf or handled in another way
        else
            out(i) = varAhead / varBehind;
        end
    end
    
    % Edge values are already set to zero
end
