clear all; close all;
% import helper functions from functions folder
addpath(genpath("functions"))

% load the weights of the neural network trained in Python
load("models/example_oscillatory.mat")

save = true;

% extract the values of the weights and biases from the neural network
number_of_decimal_places = 3;
w1 = round(first_layer_weights,number_of_decimal_places,"decimals");
b1 = round(first_layer_biases,number_of_decimal_places,"decimals");
alphas = round(output_layer_weights,number_of_decimal_places,"decimals");
hidden_nodes = length(b1);


% select a finite time-scale seperation parameter (less than 1)
time_scale = 0.01;

% select a period of time to run the ODE system
t0 = 0;
tfinal = 100;

% initialize the inital conditions of all molecular concentrations in the
% system. The inital conditions of the hidden nodes have arbitrarily been set to zero
v0 = [2; 4];
y0 = [v0; zeros(hidden_nodes,1)];

% run ODE simulation using the function ODE45 
[t,p] = ode45(@(t,y) neural_crn_2dvis(t, y, betas, gamma, time_scale, w1, alphas, b1), [t0 tfinal],y0);

% run actual van der pols system with same inital conditions 
opts = odeset('RelTol',1e-10,'AbsTol',1e-10);
[tvp,pvp] = ode45(@(t,y) non_kinitic_ode(t, y), [t0 tfinal],v0, opts);
plot(p(:,1),p(:,2), 'Color','m','LineWidth',1.5); hold on;
plot(pvp(:,1),pvp(:,2), 'Color','b','LineWidth',1.5)


plot(v0(1),v0(2),'o','MarkerFaceColor','black', 'Color','black');
plot(p(end,1), p(end,2),'o','MarkerFaceColor','m', 'Color','m');
plot(pvp(end,1), pvp(end,2),'o','MarkerFaceColor','b', 'Color','b');


ax = gca;
ax.TickDir = 'out';
xlabel('$x_1$', 'Interpreter','latex','FontSize',20);
ylabel('$x_2$', 'Interpreter','latex', 'FontSize',20);
grid on;

% Save the data to .csv
if save
    writematrix([t,p],'data/bessel_osc_RNCRN_traj.csv') 
    writematrix([tvp,pvp],'data/bessel_osc_non_kinetic_traj.csv') 
end

%% produce the vector field
x = 0:0.25:15;
[x1v,x2v] = meshgrid(x,x);


% Non-kinetic ODE dynamics at each point in the mesh
dx = 6+4*besselj(0,1.5*x1v) -x2v;
dy = x1v-4;

figure;
quiver(x1v,x2v, dx,dy); axis equal;

U = zeros(size(x1v));
V = zeros(size(x1v));
[I,J] = size(x1v);

for i = 1:I
    for j = 1:J
        dpdt = degenerate_neural_crn_2dvis(x1v(i,j), x2v(i,j), betas, gamma, w1, alphas, b1);
        U(i,j) = dpdt(1);
        V(i,j) = dpdt(2);       
    end
end

%figure; 
hold on;
quiver(x1v,x2v, U,V); axis equal;

x1s = reshape(x1v,1,[]);
x2s = reshape(x2v,1,[]);

dx1_nk = reshape(dx,1,[]);
dx2_nk = reshape(dy,1,[]);

dx1_cherr_net = reshape(U,1,[]);
dx2_cherr_net = reshape(V,1,[]);

if save
    writematrix([x1s',x2s', dx1_nk', dx2_nk'],'data/bessel_osc_non_kinetic_dynamics.csv') 
    writematrix([x1s',x2s', dx1_cherr_net', dx2_cherr_net'],'data/bessel_osc_RNCRN_dynamics.csv') 
end


