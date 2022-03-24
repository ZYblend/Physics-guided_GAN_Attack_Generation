%% Runing model
% Content:
%         1. load system matrices
%         2. Check observability and controllabiltiy
%         3. Get T-horizon system matrices
%         4. Get observer gains for L2 observer
%         5. Pole placement for gain Control design
%         6. Simulation Initialization
%         7. Runing parameters (time frame...)
%
% Yu Zheng, RASLab, FAMU-FSU College of Engineering, Tallahassee, 2021, Aug.

%% 1. load system matrices
T_sample = 0.01;   % sample time step
A_bar_d = readmatrix('A_bar_d.csv');
B_bar_d = readmatrix('B_bar_d.csv');
C_obsv_d = readmatrix('C_obsv_d.csv');
D_obsv_d = readmatrix('D_obsv_d.csv');

[n_states,n_int] = size(B_bar_d);
n_meas = size(C_obsv_d,1);

% critical measurement matrix
Cm = ones(1,n_states);
n_critical = size(Cm,1);


% %% 2. Check observability and controllabiltiy
% disp('controllability')
% disp(rank(ctrb(A_bar_d,B_bar_d))) % fully controllable with PID controller
% disp('observability')
% disp(rank(obsv(A_bar_d,C_obsv_d))) % fully observable


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
% disp('discrete controller (A-B*K) eigenvalues: less than 1?')
% disp(eig(A_bar_d-B_bar_d*K).')


%% 6. Simulation Initialization
% state initialization
x0          = 0.3+ 0.2*rand(n_states,1);
x0_hat      = zeros(n_states,1);
xd          = 0.5*ones(n_states,1);

yc_d = Cm*xd;

% Delay tape initialization
X0          = zeros(n_states,T);
Y0          = zeros(n_meas,T);
U0          = zeros(n_int,T+1);

offset = -inv(B_bar_d)*(A_bar_d-eye(n_states))*xd;


%% 7. runing parameters
N_samples      = 800;                % The total number of samples to run
T_final        = N_samples*T_sample;  % Total time for simulation (20s)
T_start_attack = 0.1*T_final;         % start injecting attack at 10s
T_start_opt(:)    = 1.5*T*T_sample;   % start state estimation at 
T_stop_attack = T_final;        % stop injecting attack at 10s