function [a,exitflag] = get_contact_radius(b,h,R,t)
% GET_CONTACT_RADIUS is a function invoked by the master AFM post-processor
%   code in order to calculate the contact radius of the blunt cone
%   geometry. The contact radius is derived from Equation (4) of Briscoe et
%   al. (Phys. D: Appl. Phys. 27, 1994. pg 1158) and is numerically
%   determined using nonlinear Gauss-Newton approximation.
%
%   [A,CONDITION] = GET_CONTACT_RADIUS(B,H,R,T) gives the contact radius
%   'a' for a given defect size 'b', depth 'h', radius of curvature 'r' and
%   cone angle 't'. For the standard blunt-cone equivalent of a pyramidal
%   AFM tip R = 50nm and t = 37.5 degrees. CONDITION is the exit flag for
%   the solvers iteration engine. For more information on exit conditions
%   see "exitflag" under FSOLVE.
%
%   AZELOGLU E.U. (c) 2005 
%   Modified by Jonathan C. Haydak 2020
% 
h = double(h);

if sqrt(R * h) <= b.^2
    warning('The requested depth is within the defect zone!');
end

guess = 0.5*b;
options = optimset('display','off','TolFun',1e-12);
F = @(a) contact_expression(a,b,h,R,t);
[a,fval,exitflag] = fsolve(F,guess,options);
%[a,fval,exitflag] = fminbnd(F,1e-15,1e5,options);
%a = abs(a);
% a = lsqnonlin(@contact_expression,guess,b,[],options,b,h,R,t);

% if abs(fval) > .01 || abs(real(a)-1e5) <10
%     5;
% end
% 
% if h > 500
%     5;
% end

a = real(a);

end

function F = contact_expression(a,b,h,R,t)
F = (h + (a./R).*(sqrt(a.^2 - b.^2) - a) ...
    -(a./tan(t)).* (0.5*pi - asin(b./a)));
%F = abs(double(F));
end