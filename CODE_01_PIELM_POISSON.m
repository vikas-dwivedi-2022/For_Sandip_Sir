clc; clear;close all;
%% User Input
NN=2000;  % Number of Neurons
exact=@(x,y)0.25*(1-x.^2-y.^2); % Exact Solution
fr=@(x,y) -1+0*x+0*y; % RHS of Poisson Equation

%% Create Geometry
X_pde=readmatrix('PDE_Points.csv');
X_bc=readmatrix('BC_Points.csv');
X_pde=[X_pde ones(size(X_pde(:,1)))];
X_bc=[X_bc ones(size(X_bc(:,1)))];

N=length(X_pde);

%% Random Weights and Biases
p=rand(NN,1);q=rand(NN,1);s=rand(NN,1);
w=[p,q,s];

%% Equations from PDE 
RHS_f = fr(X_pde(:,1),X_pde(:,2));
LHS_f = get_d2phi(X_pde*w') .*repmat((p.*p)',N,1)+get_d2phi(X_pde*w').*repmat((q.*q)',N,1);

%% Equations from BC 
RHS_bc = exact(X_bc(:,1),X_bc(:,2));
LHS_bc = get_phi(X_bc*w');

%% Assembly 
LHS=[LHS_f;LHS_bc];
RHS=[RHS_f;RHS_bc];

%% Pseudo-Inverse
tic;
c=pinv(LHS)*RHS;

%% Prediction with Optimized Weights
u = get_phi(X_pde*w')*c;
toc;

%% L2 Error
error=exact(X_pde(:,1),X_pde(:,2))-u;
L2=sqrt(sum(error.^2));

%% Plots

% PIELM Solution and Error
figure;
%-------------------------------------------------------------------------------------------%
fig=gcf;
tri = delaunay(X_pde(:,1),X_pde(:,2));
h = trisurf(tri, X_pde(:,1),X_pde(:,2), u);
axis vis3d
xlabel('x','interpreter','latex','FontSize',10)
ylabel('y','interpreter','latex','FontSize',10)
title(sprintf(['PIELM Solution: ','L2 Error = %0.1e'],L2),'Interpreter','latex','FontSize',10)
shading interp;
colorbar 
grid off
view(2)
% exportgraphics(fig, 'PIELM_2000.eps', 'ContentType', 'vector');
exportgraphics(fig, 'FIG_03_PIELM_2000.png', 'Resolution', 300);
%-------------------------------------------------------------------------------------------%
figure;
fig=gcf;
tri = delaunay(X_pde(:,1),X_pde(:,2));
h = trisurf(tri, X_pde(:,1),X_pde(:,2), exact(X_pde(:,1),X_pde(:,2)));
axis vis3d
xlabel('x','interpreter','latex','FontSize',10)
ylabel('y','interpreter','latex','FontSize',10)
title('True Solution: $u(x,y) = (1-x^2-y^2)/4$','interpreter','latex','FontSize',10)
shading interp;
colorbar
grid off
view(2)
% exportgraphics(fig, 'True_Sol.eps', 'ContentType', 'vector');
exportgraphics(fig, 'FIG_02_True_Sol.png', 'Resolution', 300);
%-------------------------------------------------------------------------------------------%
% Sampling Points
figure()
fig=gcf;
scatter(X_pde(:,1), X_pde(:,2), 5, 'filled', 'Marker', 'o')
axis square
xlabel('x','interpreter','latex','FontSize',20)
ylabel('y','interpreter','latex','FontSize',20)
title('Sampling Points','interpreter','latex','FontSize',20)
exportgraphics(fig, 'FIG_01_Sampling_Points.png', 'Resolution', 300);
% close all;
%-------------------------------------------------------------------------------------------%
%                                        Subroutines                                        %
%-------------------------------------------------------------------------------------------%
%1 Activation
function phi = get_phi(z)
phi = tanh(z);
end
%2 Derivative of Activation
function dphi = get_dphi(z)
dphi = (sech(z)).^2;
end
%3 Second Derivative of Activation
function d2phi=get_d2phi(z)
d2phi=-2*tanh(z).*((sech(z)).^2);
end

