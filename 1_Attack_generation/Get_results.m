%% This file is to run one generated attack dataset through the water distribution ssytem and prepare attack detection training dataset
% content:
%         1. Prepare simulation parameters
%         2. Run simulation
%         3. Generate training dataset:
%                         1) Load simulation resulting data
%                         2) Calculate Deviation ratio from nominal status: 
%                                                     (1) Detection(estimation) Residual; (2) critical measurement
%                         3) label attack dataset by checking deviations with thresholds
%                                       (1) successful attack: deviation(detection residual) < thresh1 "And" deviation(critical meas) > thresh2
%                                       (2) Label "attack"   : deviation(detection residual) > thresh1 "Or"  deviation(critical meas) > thresh2
%                            Explanation: Successful attack can bypass detection and arouse enough bias on critical measurements
%                                          But we need to train attack detection algorithm to identify not only successful attack but also
%                                           un-successful attacks
%         4. Save dataset:
%                         Input (T*n_meas+T*n_int -by- 1): [current measurement [i]; (T-1) measurement history [i-T+1:i-1]; input history in window [i-T:i-1] ]
%                                   where n_meas is measurement dimension, n_int is input dimension.
%                         Output (n_meas-by-1)           : attack (safe node) localization at current time step [i] (0:attack, 1: safe)
%         5. Plotting: show the power of attacks in the simulation 
%                        (Notice: when run in a for loop to obatin multiple training dataset, please comment this part to save runing time)
%
% Yu Zheng, RASLab, FAMU-FSU College of Engineering, Tallahassee, 2021, Aug.

% clear variables
% close all
% clc

%% load attacks and prepare system parameters
Run_model_without_attackDetection

%% Run simulink model
simOut = sim('WDS');

%% Generate training dataset
N_start_attack = T_start_attack/T_sample;
N_attack = N_samples-N_start_attack+1;
% load simulation result
x = simOut.logsout.getElement('x').Values.Data;

y_attacked      = simOut.logsout.getElement('y_attacked').Values.Data; % attacked measurements
y_attacked = y_attacked(N_start_attack:N_samples,:);

u_attacked      = simOut.logsout.getElement('u_attacked').Values.Data;
u_attacked = u_attacked(N_start_attack:N_samples,:);

r               = simOut.logsout.getElement('r').Values.Data;  
r_attacked      = simOut.logsout.getElement('r_attacked').Values.Data;
yc               = simOut.logsout.getElement('yc').Values.Data;
yc_attacked      = simOut.logsout.getElement('yc_attacked').Values.Data;


save x_nominal.mat x
save r_nominal.mat r

% calculate performance criterion
Deviation_residual_full = 100*abs(r-r_attacked)./r;
Deviation_residual = Deviation_residual_full(N_start_attack+1:end);
Deviation_critical_meas_full = 100*abs(yc-yc_attacked)./yc;
Deviation_critical_meas = Deviation_critical_meas_full(N_start_attack+1:end);

% yc_max_delta = max(abs(yc_delta-yc_delta0));
% Deviation_critical_meas_full = abs(yc_delta-yc_delta0)/yc_max_delta;
% Deviation_critical_meas = Deviation_critical_meas_full(N_start_attack+1:end);
% 
% r_max_delta = max(abs(r_delta-r_delta0));
% Deviation_residual_full = abs(r_delta-r_delta0)/r_max_delta;
% Deviation_residual = Deviation_residual_full(N_start_attack+1:end);


% Assign attack support
tolerance = 0.01;

y_attacked_without_deadzone = y_attacked(T:end,:); % start from the N_start_attack+T, because the first T attacks cannot be tested
Q = zeros(size(y_attacked_without_deadzone));      % output
success = zeros(size(y_attacked_without_deadzone,1),1);

train_input_y = zeros(size(y_attacked_without_deadzone,1),n_meas*T);
train_input_u = zeros(size(y_attacked_without_deadzone,1),n_int*T);

for t_iter = T:N_samples-N_start_attack+1
    q = ones(1,n_meas);
    if Deviation_residual(t_iter) <= Tau2_real + tolerance && Deviation_critical_meas(t_iter) >= Tau1_real - tolerance  % successful attacks
%         q(I_attack_local) = 0;    % 0: attack, 1: no attack
        success(t_iter-T+1) = 1;
    end
    if Deviation_critical_meas(t_iter) >= Tau1_real - tolerance || Deviation_residual(t_iter) >= Tau2_real + tolerance  % successful attacks
        q(I_attack_local) = 0;    % 0: attack, 1: no attack
    end
    Q(t_iter-T+1,:) = q;
    
    train_input_y_temp = y_attacked(t_iter-T+1:t_iter,:).';
    train_input_y(t_iter-T+1,:) = train_input_y_temp(:).';
    
    train_input_u_temp = u_attacked(t_iter-T+1:t_iter,:).';
    train_input_u(t_iter-T+1,:) = train_input_u_temp(:).';
end

train_input = [train_input_y,train_input_u];

train_data = {train_input,Q};
        
%% save training dataset
% save train_dataset.mat train_data
% success_matrix = [success, Deviation_residual(T:end), Deviation_critical_meas(T:end)];
% writematrix(success_matrix,'success.txt');


%% plotting
% figure (1)
% subplot(1,2,1)
% plot(Deviation_residual);  % Deviation from the nominal residual
% hold on, plot(Tau2_real*ones(size(r)));
% xlabel('Time')
% ylabel('Deviation Ratio (%) from nominal Residual')
% xlim([0,720])
% 
% subplot(1,2,2)
% plot(Deviation_critical_meas);  % Deviation from the nominal critical measurement
% hold on, plot(Tau1_real*ones(size(r)));
% xlabel('Time')
% ylabel('Deviation Ratio (%) from nominal critical measurement')
% xlim([0,720])




