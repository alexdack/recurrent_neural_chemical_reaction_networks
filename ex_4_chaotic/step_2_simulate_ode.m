clear all; close all;
% import helper functions from functions folder
 addpath(genpath("functions"))

% load the weights of the neural network trained in Python
load("models/example_hasting_powell.mat");

% extract the values of the weights and biases from the neural network
number_of_decimal_places = 3;
w1 = round(first_layer_weights,number_of_decimal_places,"decimals");
b1 = round(first_layer_biases,number_of_decimal_places,"decimals");
alphas = round(output_layer_weights,number_of_decimal_places,"decimals");
hidden_nodes = length(b1);


% initialize the values of offest parameters used in the training (these
% should be the same as in the Python code)
gamma = 1;
beta = [0; 0; 0];



% select a finite time-scale seperation parameter (less than 1)ref_chaos
time_scale_1 = 0.1;

% select a period of time to run the ODE system
t0 = 0;
tFinal = 1000;

%%
close all;
v0 = [0.25; 0.25; 0.25];
y0 = [v0; zeros(hidden_nodes,1)];

% run ODE simulation using the function ODE45 
[t,p] = ode45(@(t,y) neural_crn_3dvis(t, y, gamma, beta, time_scale_1, w1, alphas, b1), [t0 tFinal],y0);

figure; subplot(1,2,2); hold on;
plot3(p(:,1), p(:,2), p(:,3), 'Color','m' )
view(3);
xlim([0,1])
ylim([0,1])
zlim([0,0.5])
xlabel('$x_1$', 'Interpreter','latex');
ylabel('$x_2$', 'Interpreter','latex');
zlabel('$x_3$', 'Interpreter','latex');


%%  Chaotic system
subplot(1,2,1); 
r= 2.5;
k=1.5;
a1 = 4.0;
a2 = 4.0;
b1 = 3.0;
b2 = 3.0;
d1 = 0.4;
d2 = 0.6;

fun = @(t,x) non_kinetic_ode(t, x, r, k, a1, a2, b1, b2, d1, d2);
[tt, x_t] = ode45(fun, [0 tFinal], v0);

plot3(x_t(:,1), x_t(:,2), x_t(:,3), 'Color','b');
xlabel("$x_1$", 'Interpreter','latex')
ylabel("$x_2$", 'Interpreter','latex')
zlabel("$x_3$", 'Interpreter','latex')
xlim([0,1])
ylim([0,1])
zlim([0,0.5])

%%
figure; 
subplot(3,1,1); plot(t, p(:,1),'Color','m'); hold on;
plot(tt, x_t(:,1), 'Color','b');
xlim([0,300])
ylim([0,1])

subplot(3,1,2); plot(t, p(:,2),'Color','m'); hold on;
plot(tt, x_t(:,2), 'Color','b');
xlim([0,300])
ylim([0,1])

subplot(3,1,3); plot(t, p(:,3),'Color','m'); hold on;
plot(tt, x_t(:,3), 'Color','b');
xlim([0,300])
ylim([0,1])



%% Save the data to .csv
writematrix([t,p],'data/hasting_powell_RNCRN_traj.csv') 
writematrix([tt,x_t],'data/hasting_powell_non_kinetic_traj.csv') 
