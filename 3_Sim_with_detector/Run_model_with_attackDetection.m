%% Run model for simulation (with attack detection and localization)
% Description: This file is to prepare simulation parameters 
%              In this simulation, four scenarios are compared:
% Content:
%         1. load system
%         2. parameters for sample-based moving-horizon FDIA
%         3. Parameters for MLPs and Pruning algorithm
%
% Yu Zheng, RASLab, FAMU-FSU College of Engineering, Tallahassee, 2021, Aug.


% add path
currentpath = pwd;
motherpath = erase(currentpath,"\3_Sim_with_detector");
addpath(append(motherpath,'\1_Attack_generation'));
addpath(append(motherpath,'\2_MLP_training'));
addpath(append(motherpath,'\Common_fcns'));


%% water tank system model matrices
T_sample = 0.01;

A_bar_d = readmatrix('A_bar_d.csv');
B_bar_d = readmatrix('B_bar_d.csv');
C_obsv_d = readmatrix('C_obsv_d.csv');
D_obsv_d = readmatrix('D_obsv_d.csv');


[n_states,n_int] = size(B_bar_d);
n_meas = size(C_obsv_d,1);

Cm = ones(1,n_states);
n_critical = size(Cm,1);

k0 = 42;   % lower bound of number of measurements for which the system remain full observability

%% system matrix unit testing
%controllability and observability
% disp('controllability')
% disp(rank(ctrb(A_bar_d,B_bar_d))) % fully controllable with PID controller
% disp('observability')
% disp(rank(obsv(A_bar_d,C_obsv_d))) % fully observable


%% T time horizon
% T = round(2*n_states);
T = 8;
[H0_full,H1_full,F] = opti_params(A_bar_d,B_bar_d,C_obsv_d,T);
% H0: state-ouput linear map                      [n_meas*T-by-n_states]
% H1:  input-output linear map                     [n_meas*T-by-n_int*(T-1)]
% F: Observer input-state propagation matrix      [n_meas-by-n_int*(T-1)]


%% Observer Dynamics for attack-free case
H0_full_pinv = pinv(H0_full,0.001);
Ly = A_bar_d.'*H0_full_pinv;
Lu = F-A_bar_d.'*H0_full_pinv*H1_full;
% A_T = A_bar_d^T;

% residual
H0_full_pinv = pinv(H0_full,0.001);
H0_perp_full = eye(size(H0_full,1)) - H0_full*H0_full_pinv;


%% Gain Controller
Pc = linspace(0.1,0.2, n_states);
K = place(A_bar_d,B_bar_d,Pc);
% disp('discrete controller (A-B*K) eigenvalues: less than 1?')
% disp(eig(A_bar_d-B_bar_d*K).')


%% Simulation Initialization
% state initialization
x0          = 0.3+ 0.2*rand(n_states,1);
x0_hat      = zeros(n_states,1);
xd          = 0.5*ones(n_states,1);

yc_d = Cm*xd;

% Delay tape initialization
Q0          = ones(n_meas,T);    % initial support: all safe
Y0          = zeros(n_meas,T);
U0          = zeros(n_int,T+1);

offset = -inv(B_bar_d)*(A_bar_d-eye(n_states))*xd;

%% runing parameters
N_samples      = 800;                % The total number of samples to run
T_final        = N_samples*T_sample;  % Total time for simulation (20s)
T_start_attack = 0.1*T_final;         % start injecting attack at 10s
T_start_opt(:)    = 1.5*T*T_sample;
T_start_detect = T_start_attack+T*T_sample;

N_start_attack = T_start_attack/T_sample;
N_attack = N_samples-N_start_attack+1;

% load nominal states and residual
x_nominal = load('x_nominal.mat').x;
r_nominal = load('r_nominal.mat').r;
time_interv_full = linspace(0,T_final,N_samples+1);
x_ts = timeseries(x_nominal,time_interv_full);
r_ts = timeseries(r_nominal,time_interv_full);

% %% 2. sample-based moving-horizon FDIA
% % load data
% n_attack = round(length(I_attack)/T);   % number of attacks
% I_attack_local = I_attack(1:n_attack)+ones(n_attack,1);  % attacks upport for one time step
%                                                          % notice: python index starts from 0
M1 = readmatrix('M1.csv');
M2 = readmatrix('M2.csv');

% thresholds for detection and effectiveness calculated by M1 and M2
tau1 = readmatrix('tau1.csv');
tau2 = readmatrix('tau2.csv');

% thresholds for detection and effectiveness in the system simulation
Thresh1 = 2;   % for effect
Thresh2 = 1.5;   % for detection

% 
% % choose one best T attacks (y1_effect is biggest, y2_detect<tau2)
% % the T attacks will be injected at the beginning T time steps
% [~,I_best] = max(y1_effect(y2_detect<=tau2));
% attack_potential = attack(y2_detect<=tau2,:);
% attack_0 = attack_potential(I_best,:);
% 
% % get attacks in row-wise (each row correponds to one attack)
% % atatck(m-by-n_attack*T) -- attack_tank (m*T -by- n_attack)
% attack_tank = zeros(size(attack,1)*T,n_attack);
% E0 = zeros(T,n_attack);
% for index = 1:T
%     attack_tank((index-1)*size(attack,1)+1:index*size(attack,1),:) = attack(:,(index-1)*n_attack+1:index*n_attack);
%     E0(index,:) = attack_0(:,(index-1)*n_attack+1:index*n_attack);
% end
% 
% % get attacks in full dimension
% % attack_tank (m*T -by- n_attack) -- attack_tank_full (m*T-by-n_meas)
% attack_tank_full = zeros(size(attack_tank,1),n_meas);
% E0_full = zeros(T,n_meas);
% for iter = 1:n_attack
%     index = I_attack_local(iter);
%     attack_tank_full(:,index) = attack_tank(:,iter);
%     E0_full(:,index) = E0(:,iter);
% end

%% MLP and Pruning
% MLP1
datapath = append(motherpath,'\2_MLP_training\');

w1_1 = readmatrix(append(datapath,'\MLP1\w1.csv'));  % [input_size,hidenlayer1_size]
w2_1 = readmatrix(append(datapath,'MLP1\w2.csv'));  % [hidenlayer1_size,hidenlayer2_size]
w3_1 = readmatrix(append(datapath,'MLP1\w3.csv'));  % [hidenlayer2_size,hidenlayer3_size]
w4_1 = readmatrix(append(datapath,'MLP1\w4.csv'));  % [hidenlayer3_size,Output_size]
b1_1 = readmatrix(append(datapath,'MLP1\b1.csv'));  % [hidenlayer1_size,1]
b2_1 = readmatrix(append(datapath,'MLP1\b2.csv'));  % [hidenlayer2_size,1]
b3_1 = readmatrix(append(datapath,'MLP1\b3.csv'));  % [hidenlayer3_size,1]
b4_1 = readmatrix(append(datapath,'MLP1\b4.csv'));  % [Output_size,1]

p_MLP1 = load('P1.mat').P1;

% MLP2 (With automated attack generation)
w1_2 = readmatrix(append(datapath,'\MLP2\w1.csv'));  % [input_size,hidenlayer1_size]
w2_2 = readmatrix(append(datapath,'MLP2\w2.csv'));  % [hidenlayer1_size,hidenlayer2_size]
w3_2 = readmatrix(append(datapath,'MLP2\w3.csv'));  % [hidenlayer2_size,hidenlayer3_size]
w4_2 = readmatrix(append(datapath,'MLP2\w4.csv'));  % [hidenlayer3_size,Output_size]
b1_2 = readmatrix(append(datapath,'MLP2\b1.csv'));  % [hidenlayer1_size,1]
b2_2 = readmatrix(append(datapath,'MLP2\b2.csv'));  % [hidenlayer2_size,1]
b3_2 = readmatrix(append(datapath,'MLP2\b3.csv'));  % [hidenlayer3_size,1]
b4_2 = readmatrix(append(datapath,'MLP2\b4.csv'));  % [Output_size,1]

p_MLP2 = load('P2.mat').P2;


% Pruning
eta = 0.7;
