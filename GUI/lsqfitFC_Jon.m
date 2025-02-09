function [yerr, soln ,yp] = lsqfitFC_Jon(ext,defl,xcndx)
% % Here I am spliting up the curve into pre- and post- contact parts.
% A linear and second order polynomial part.


x1 = ext(1:xcndx);
y1 = defl(1:xcndx);
x2 = ext(xcndx:end) - x1(end);
y2 = defl(xcndx:end);
xc = ext(xcndx);
% % Fit using all points

x1len = length(x1);
x2len = length(x2);
A = zeros(x1len+x2len,4);
A(1:x1len,:) = [ones(x1len,1), x1, zeros(x1len,1), zeros(x1len,1)];
A(x1len+1:end,:) = [ ones(x2len,1), xc*ones(x2len,1), x2, x2.^2];
b = [y1; y2];


% Solve using normal equation

soln = transpose(A)*A \ transpose(A)*b;

% % Calculate error on the solution and plot it

a=soln(1); b=soln(2);c = soln(3); d = soln(4);
y1p = a+b*x1;
y2p = a + b*xc+c*x2+d*x2.^2;
% 
% figure
% plot([x1; x2+x1(end)],[y1;y2],'b')
% hold on
% plot([x1;x2+x1(end)],[y1p;y2p],'r')
% legend('Data','Fit')

yerr = [y1p(1:end-1) - y1(1:end-1); y2p-y2];
yerr = sum(norm(yerr));

yp = [y1p(1:end-1); y2p];


end