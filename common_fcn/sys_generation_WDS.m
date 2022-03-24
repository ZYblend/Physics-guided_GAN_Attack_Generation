% water distribution system generation
% This is file is to generate discrete system model for water distribution system
%  x_{i+1} = A_bar_d * x_{i} + B_bar_d * u_{i}
%  y_{i}   = C_obsv_d * x_{i} + D_obsv_d * u_{i}
%  Once the system matrices are generated, they will be used thoughout all simulations
%
%  Please refer to paper <Algorithm Design for Resilient Cyber-Physical Systems using Automated Attack Generative Models> 
%   for detailed system description
%
% Yu Zheng, RASLab, FAMU-FSU College of Engineering, Tallahassee, 2021, Aug.

clear all
clc

%% sizes
n_states = 10;  % 10 states
n_int = 10;     % 10 inputs
               
%% System dynamics
% A
d1_min = -0.5;
d1_max = -0.6;
d2 = 0.187;
d = d1_min+(d1_max-d1_min)*rand(n_states,1);
A_bar = diag(d) + diag(d2*ones(n_states-1,1),-1);

% B
b1_min = 0.5;
b1_max = 1;
b2 = -0.845;
b = b1_min+(b1_max-b1_min)*rand(n_int,1);
B_bar = (diag(b) + diag(b2*ones(n_int-1,1),-1)).';

% C
% water level difference (two tanks)
C_temp1 = diag(ones(n_states-1,1),-1).';  % joint water lavel difference
C_temp2 = diag(ones(n_states-1,1),-2).';  % gap 1 water level difference
C_temp3 = diag(ones(n_states-1,1),-3).';  % gap 2 water level difference
C_temp4 = diag(ones(n_states-1,1),-4).';  % gap 3 water level difference
% water level difference (three tanks)
C_temp5 = C_temp1(1:n_states-2,1:n_states)+C_temp2(1:n_states-2,1:n_states);
C_temp6 = C_temp2(1:n_states-3,1:n_states)+C_temp3(1:n_states-3,1:n_states);
C_temp7 = C_temp3(1:n_states-4,1:n_states)+C_temp4(1:n_states-4,1:n_states);
% water level sum 
C_temp8 = 10 * rand(9,10);

C_obsv = [eye(n_states);                                                % water level of 10 tanks
          eye(n_states-1,n_states)- C_temp1(1:n_states-1,1:n_states);   % 9
          eye(n_states-2,n_states)- C_temp2(1:n_states-2,1:n_states);   % 8
          eye(n_states-3,n_states)- C_temp3(1:n_states-3,1:n_states);   % 7
          eye(n_states-4,n_states)- C_temp4(1:n_states-4,1:n_states);   % 6
          eye(n_states-2,n_states)- C_temp5;
          eye(n_states-3,n_states)- C_temp6;
          eye(n_states-4,n_states)- C_temp7]; 
% C_obsv = 10*rand(40,10);

n_meas = size(C_obsv,1);  % 40 measurements

% D 
D_obsv = zeros(n_meas,n_int); 


%% Discretized System Model
T_sample = 0.01;
[A_bar_d, B_bar_d] = discretize_linear_model(A_bar,B_bar,T_sample);
C_obsv_d = C_obsv;
D_obsv_d = D_obsv;

disp('eigenvalues of linearized A')
disp(eig(A_bar_d).')


%% Save system matrices
writematrix(A_bar_d,'A_bar_d.csv');
writematrix(B_bar_d,'B_bar_d.csv');
writematrix(C_obsv_d,'C_obsv_d.csv');
writematrix(D_obsv_d,'D_obsv_d.csv');

%% controllability and observability test
disp('controllability')
disp(rank(ctrb(A_bar_d,B_bar_d))) % fully controllable with PID controller
disp('observability')
disp(rank(obsv(A_bar_d,C_obsv_d))) % fully observable