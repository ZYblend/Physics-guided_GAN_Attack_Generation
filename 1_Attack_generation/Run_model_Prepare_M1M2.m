%% Run system and Get M1, M2 matrix for system
% This file is to prepare parameters for water distribution system
%  and prepare M1, M2 matrix for attack generator
%    1. load system matrices
%    2. Check observability and controllabiltiy
%    3. Get T-horizon system matrices
%    4. Get observer gains for L2 observer
%    5. Pole placement for gain Control design
%    6. Get M1, M2 matrix
%
% Yu Zheng, RASLab, FAMU-FSU College of Engineering, Tallahassee, 2021, Aug.

clear variables
close all
clc

% add path
currentpath = pwd;
motherpath = erase(currentpath,"\1_Attack_generation");
addpath(append(motherpath,'\Common_fcns'));

%% 1. load system matrices
T_sample = 0.01;   % sample time step
A_bar_d = readmatrix('A_bar_d.csv');
B_bar_d = readmatrix('B_bar_d.csv');
C_obsv_d = readmatrix('C_obsv_d.csv');
D_obsv_d = readmatrix('D_obsv_d.csv');

[n_states,n_int] = size(B_bar_d);
n_meas = size(C_obsv_d,1);

% critical measurement matrix
Cc = ones(1,n_states);
n_critical = size(Cc,1);


%% 2. Check observability and controllabiltiy
disp('controllability')
disp(rank(ctrb(A_bar_d,B_bar_d))) % fully controllable with PID controller
disp('observability')
disp(rank(obsv(A_bar_d,C_obsv_d))) % fully observable


%% 3. Get T-horizon system matrices
T = 8;
[H0,H1,F] = opti_params(A_bar_d,B_bar_d,C_obsv_d,T);
% H0: state-ouput linear map                       [n_meas*T-by-n_states]
% H1:  input-output linear map                     [n_meas*T-by-n_int*(T-1)]
% F:  Observer input-state propagation matrix      [n_meas-by-n_int*(T-1)]


%% 4. Get observer gains for L2 observer
H0_pinv = pinv(H0,0.001);
Ly = A_bar_d.'*H0_pinv;
Lu = F-A_bar_d.'*H0_pinv*H1;

% residual
H0_perp = eye(size(H0,1)) - H0*H0_pinv;


%% 5. Pole placement for gain Control design
Pc = linspace(0.1,0.2, n_states);
K = place(A_bar_d,B_bar_d,Pc);
disp('discrete controller (A-B*K) eigenvalues: less than 1?')
disp(eig(A_bar_d-B_bar_d*K).')


%% 6. Get M1, M2 matrix
[M1,M2] = Get_Transfer_matrix(A_bar_d,B_bar_d,Cc,H0,-K,T,n_meas);

writematrix(M1,'M1.csv');
writematrix(M2,'M2.csv');