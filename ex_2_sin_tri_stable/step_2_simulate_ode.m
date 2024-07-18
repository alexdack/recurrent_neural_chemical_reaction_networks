clear all; close all;
% import helper functions from functions folder
 addpath(genpath("functions"))

% load the weights of the neural network trained in Python
load("models/example_sin_tri_stable.mat")

% extract the values of the weights and biases from the neural network
number_of_decimal_places = 3;
w1 = round(first_layer_weights,number_of_decimal_places,"decimals");
b1 = round(first_layer_biases,number_of_decimal_places,"decimals");
alphas = round(output_layer_weights,number_of_decimal_places,"decimals");
hidden_nodes = length(b1);

% initialize the values of offest parameters used in the training (these
% should be the same as in the Python code)
gamma = 1;
beta_1 = 0;

% select a finite time-scale seperation parameter (less than 1)
time_scale_1 = 0.01;

% select a period of time to run the ODE system
tstart = 0;
tfinal = 30;

% initialize the inital conditions of all molecular concentrations in the
% system. The inital conditions of the hidden nodes have arbitrarily been set to zero
v0 = 7*pi/2;
v1 = 5*pi/2;
v2 = 3*pi/2;
v3 = pi/2;
v4 = 9*pi/2;
v5 = 11*pi/2;

y0 = [v0; zeros(hidden_nodes,1)];
y1 = [v1; zeros(hidden_nodes,1)];
y2 = [v2; zeros(hidden_nodes,1)];
y3 = [v3; zeros(hidden_nodes,1)];
y4 = [v4; zeros(hidden_nodes,1)];
y5 = [v5; zeros(hidden_nodes,1)];

% run ODE simulation using the function ODE45 
[t_RNCRN_0,p_RNCRN_0] = ode45(@(t,y) neural_crn_1dvis(t, y, gamma, beta_1, time_scale_1, w1, alphas, b1), [tstart tfinal],y0);
[t_RNCRN_1,p_RNCRN_1] = ode45(@(t,y) neural_crn_1dvis(t, y, gamma, beta_1, time_scale_1, w1, alphas, b1), [tstart tfinal],y1);
[t_RNCRN_2,p_RNCRN_2] = ode45(@(t,y) neural_crn_1dvis(t, y, gamma, beta_1, time_scale_1, w1, alphas, b1), [tstart tfinal],y2);
[t_RNCRN_3,p_RNCRN_3] = ode45(@(t,y) neural_crn_1dvis(t, y, gamma, beta_1, time_scale_1, w1, alphas, b1), [tstart tfinal],y3);
[t_RNCRN_4,p_RNCRN_4] = ode45(@(t,y) neural_crn_1dvis(t, y, gamma, beta_1, time_scale_1, w1, alphas, b1), [tstart tfinal],y4);
[t_RNCRN_5,p_RNCRN_5] = ode45(@(t,y) neural_crn_1dvis(t, y, gamma, beta_1, time_scale_1, w1, alphas, b1), [tstart tfinal],y5);

% run ODE simulation using the function ODE45 
[t0,p0] = ode45(@non_kinetic_ode, [tstart tfinal],v0);
[t1,p1] = ode45(@non_kinetic_ode, [tstart tfinal],v1);
[t2,p2] = ode45(@non_kinetic_ode, [tstart tfinal],v2);
[t3,p3] = ode45(@non_kinetic_ode, [tstart tfinal],v3);
[t4,p4] = ode45(@non_kinetic_ode, [tstart tfinal],v4);
[t5,p5] = ode45(@non_kinetic_ode, [tstart tfinal],v5);

% plot the resulting system trajectory 
plot(t0, p0, 'Color','b','LineWidth',1.5); hold on;
plot(t_RNCRN_0,p_RNCRN_0(:,1), 'Color','m','LineWidth',1.5, 'LineStyle','--');
plot(t1, p1, 'Color','b','LineWidth',1.5)
plot(t_RNCRN_1,p_RNCRN_1(:,1), 'Color','m','LineWidth',1.5, 'LineStyle','--')
plot(t2, p2, 'Color','b','LineWidth',1.5)
plot(t_RNCRN_2,p_RNCRN_2(:,1), 'Color','m','LineWidth',1.5, 'LineStyle','--')
plot(t3, p3, 'Color','b','LineWidth',1.5)
plot(t_RNCRN_3,p_RNCRN_3(:,1), 'Color','m','LineWidth',1.5, 'LineStyle','--')
plot(t4, p4, 'Color','b','LineWidth',1.5)
plot(t_RNCRN_4,p_RNCRN_4(:,1), 'Color','m','LineWidth',1.5, 'LineStyle','--')
plot(t5, p5, 'Color','b','LineWidth',1.5)
plot(t_RNCRN_5,p_RNCRN_5(:,1), 'Color','m','LineWidth',1.5, 'LineStyle','--')
legend('$\bar{x}_1(t)$', "$x_1(t): \mu = 0.01$",'Interpreter','latex','FontSize',15)
ax = gca;
ax.TickDir = 'out';
xlabel('time', 'Interpreter','latex','FontSize',30);
ylabel('molecular concentration', 'Interpreter','latex', 'FontSize',30);
grid on;


%% Save the data to .csv
writematrix([t_RNCRN_0,p_RNCRN_0],'data/tri_stable_RNCRN_traj_10.csv')
writematrix([t_RNCRN_1,p_RNCRN_1],'data/tri_stable_RNCRN_traj_7.csv')
writematrix([t_RNCRN_2,p_RNCRN_2],'data/tri_stable_RNCRN_traj_4.csv')
writematrix([t_RNCRN_3,p_RNCRN_3],'data/tri_stable_RNCRN_traj_1.csv')
writematrix([t_RNCRN_4,p_RNCRN_4],'data/tri_stable_RNCRN_traj_14.csv')
writematrix([t_RNCRN_5,p_RNCRN_5],'data/tri_stable_RNCRN_traj_17.csv')

writematrix([t0,p0],'data/tri_stable_non_kinetic_traj_10.csv') 
writematrix([t1,p1],'data/tri_stable_non_kinetic_traj_7.csv') 
writematrix([t2,p2],'data/tri_stable_non_kinetic_traj_4.csv') 
writematrix([t3,p3],'data/tri_stable_non_kinetic_traj_1.csv') 
writematrix([t4,p4],'data/tri_stable_non_kinetic_traj_14.csv') 
writematrix([t5,p5],'data/tri_stable_non_kinetic_traj_17.csv') 

%% compute the approximate vector field
%initialize the training domain used
lower_limit = 0;
upper_limit = 20*pi;

% select a step size to plot the dynamics (this can be different from the
% one used in training) it is just for visualization
step_size = 0.1;

% create the input domain of molecular concentrations of x_1
input_domain = lower_limit:step_size:upper_limit;

% compute the desired dynamics 
desired_dyn = sin(input_domain);

% compute the feedback for each value of the input domain
feedback_dyn = degenerate_neural_subsystem(input_domain, w1, alphas, b1, gamma);

% compute the approximate dynamics of the visble species
approx_dyn = beta_1 + feedback_dyn'.*input_domain;

% plot a comparison between the desired dynamics and the approximate
% dynamics
figure; hold on; 
plot(input_domain, desired_dyn,  'Color','b','LineWidth',1.5)
plot(input_domain, approx_dyn,  'Color','m','LineWidth',1.5,'LineStyle','--')
legend('$f_1(x_1) = sin(x)$', "$g_1(x_1)$ ", 'Interpreter','latex','FontSize',15)

ax = gca;
ax.TickDir = 'out';

xlabel('$x_1$', 'Interpreter','latex','FontSize',30);
ylabel('$dx_1 / dt$', 'Interpreter','latex', 'FontSize',30);
grid on;

writematrix([input_domain', desired_dyn'],'data/tri_stable_non_kinetic_dynam.csv') 
writematrix([input_domain', approx_dyn'],'data/tri_stable_RNCRN_dynam.csv')
 

