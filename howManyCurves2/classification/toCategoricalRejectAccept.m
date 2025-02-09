function catLabels = toCategoricalRejectAccept(Y)
% TOCATEGORICALREJECTACCEPT
%   Convert numeric or string labels into categorical with categories = ["reject","accept"].
%   E.g. numeric(0=>reject,1=>accept). 
%   If they are already that format, pass through.

    if isnumeric(Y)
        strY = strings(size(Y));
        strY(Y==0) = "reject";
        strY(Y==1) = "accept";
        catLabels = categorical(strY, ["reject","accept"]);
    elseif isstring(Y) || ischar(Y) || iscellstr(Y)
        strY = string(Y);
        strY(strY=="0") = "reject";
        strY(strY=="1") = "accept";
        catLabels = categorical(strY, ["reject","accept"]);
    elseif iscategorical(Y)
        % Ensure categories are in correct order
        if ~all(ismember(["reject","accept"], categories(Y)))
            newY = string(Y);
            newY(newY=="0") = "reject";  newY(newY=="1") = "accept";
            catLabels = categorical(newY, ["reject","accept"]);
        else
            catLabels = Y;
        end
    else
        error('Labels must be numeric, string, char, or categorical. Got: %s', class(Y));
    end
end
