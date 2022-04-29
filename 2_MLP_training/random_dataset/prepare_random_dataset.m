%% random attacks dataset generation
% Content:
%         1. Run model parameters
%         2. attack generation, test and prepare training dataset
%
% Yu Zheng, RASLab, FAMU-FSU College of Engineering, Tallahassee, 2021, Aug.

%% 1. system load
clear variables
close all
clc
% add path
currentpath = pwd;
motherpath = erase(currentpath,"\2_Training_dataset_gen");
addpath(append(motherpath,'\Common_fcns'));

Run_model

% thresholds for deviation ratio in real system simulation
Tau1_real = 2;  % for effect
Tau2_real = 1.5;  % for detection


%% 2. attack generation and test and prepare training dataset
nn_attack = round(linspace(1,n_meas/2,n_meas/2));    % the number of attack from 1 to n_meas/2
attack_strength = 0.5;                                 % amplifier on attack magtitude

Train_data = [];
Success_cell = [];
n_iteration =5;                               % number of dataset under different attack percentage
for Iter = 1:n_iteration
    for iter = 1:n_meas/2
        % attack support
        n_attack = nn_attack(iter);
        rng('shuffle');
        I_attack_local = randperm(n_meas,n_attack);
        I_attack =  zeros(n_attack*T,1);
        for iter2 = 1:T
            I_attack((iter2-1)*n_attack+1:iter2*n_attack) = I_attack_local+(iter2-1)*n_meas;
        end
        I_attack = sort(I_attack);
        
        % attack signal
        N_start_attack = T_start_attack/T_sample;
        N_attack = N_samples-N_start_attack+1;
        attack = attack_strength*rand(N_attack,n_attack);

        attack_full = zeros(size(attack,1),n_meas);
        for iter3 = 1:n_attack
            attack_full(:,I_attack_local(iter3)) = attack(:,iter3);
        end

        % create time-series attack vectors
        time_interv = linspace(T_start_attack,T_final,N_samples-N_start_attack+1);
        attack_ts = timeseries(attack_full,time_interv);

        % run simulink model
        simOut = sim('WDS_random');

        % extract out data
        y_attacked      = simOut.logsout.getElement('y_attacked').Values.Data; % attacked measurements
        y_attacked = y_attacked(N_start_attack:N_samples,:);

        u_attacked      = simOut.logsout.getElement('u_attacked').Values.Data;
        u_attacked = u_attacked(N_start_attack:N_samples,:);

        r               = simOut.logsout.getElement('r').Values.Data;  
        r_attacked      = simOut.logsout.getElement('r_attacked').Values.Data;
        yc               = simOut.logsout.getElement('yc').Values.Data;
        yc_attacked      = simOut.logsout.getElement('yc_attacked').Values.Data;
        
        % calculate performance criterion
%         r_delta0 = r-r;
%         r_delta = r_attacked-r;
%         yc_delta0 = yc - yc_d;
%         yc_delta = yc_attacked - yc_d;
%         yc_max_delta = max(abs(yc_delta-yc_delta0));
%         Deviation_critical_meas_full = abs(yc_delta-yc_delta0)/yc_max_delta;
%         Deviation_critical_meas = Deviation_critical_meas_full(N_start_attack+1:end);
% 
%         r_max_delta = max(abs(r_delta-r_delta0));
%         Deviation_residual_full = abs(r_delta-r_delta0)/r_max_delta;
%         Deviation_residual = Deviation_residual_full(N_start_attack+1:end);
        Deviation_residual_full = 100*abs(r-r_attacked)./r;
        Deviation_residual = Deviation_residual_full(N_start_attack+1:end);
        Deviation_critical_meas_full = 100*abs(yc-yc_attacked)./yc;
        Deviation_critical_meas = Deviation_critical_meas_full(N_start_attack+1:end);

        figure (1)
        subplot(1,2,1)
        plot(Deviation_residual);  % Deviation from the nominal residual
        hold on, plot(Tau2_real*ones(size(r)));
        xlabel('Time')
        ylabel('Deviation Ratio (%) from nominal Residual')
        xlim([0,720])
        
        subplot(1,2,2)
        plot(Deviation_critical_meas);  % Deviation from the nominal critical measurement
        hold on, plot(Tau1_real*ones(size(r)));
        xlabel('Time')
        ylabel('Deviation Ratio (%) from nominal critical measurement')
        xlim([0,720])

        % % Assign attack support
        tolerance = 0.01;

        y_attacked_without_deadzone = y_attacked(T:end,:); % start from the N_start_attack+T, because the first T attacks cannot be tested
        Q = zeros(size(y_attacked_without_deadzone));
        success = zeros(size(y_attacked_without_deadzone,1),1);

        train_input_y = zeros(size(y_attacked_without_deadzone,1),n_meas*T);
        train_input_u = zeros(size(y_attacked_without_deadzone,1),n_int*T);

        for t_iter = T:N_samples-N_start_attack+1
            q = ones(1,n_meas);
            if Deviation_residual(t_iter) <= Tau2_real + tolerance && Deviation_critical_meas(t_iter) >= Tau1_real - tolerance  % successful attacks
                success(t_iter-T+1) = 1;
            end
            if Deviation_critical_meas(t_iter) >= Tau1_real - tolerance || Deviation_residual(t_iter) > Tau2_real + tolerance  % successful attacks
                q(I_attack_local) = 0;    % 0: attack, 1: no attack
            end
            Q(t_iter-T+1,:) = q;

            train_input_y_temp = y_attacked(t_iter-T+1:t_iter,:).';
            train_input_y(t_iter-T+1,:) = train_input_y_temp(:).';

            train_input_u_temp = u_attacked(t_iter-T+1:t_iter,:).';
            train_input_u(t_iter-T+1,:) = train_input_u_temp(:).';
        end

        train_input = [train_input_y,train_input_u];
        train_data = [train_input,Q];

        Train_data{end+1}= train_data;
        success_matrix = [success, Deviation_residual(T:end), Deviation_critical_meas(T:end)];
        Success_cell{end+1} = success_matrix;
    end
    txt = ['end of iteration', num2str(Iter)];
    disp(txt);
end

%% Mix dataset and save
Train_data_matrix = cell2mat(Train_data');       % put training dataset in one matrix
I_mix = randperm(size(Train_data_matrix,1));        % mixed index

Train_data_mixed = Train_data_matrix(I_mix,:);    % mix
save Random_train_data.mat Train_data_mixed

Success_matrix =  cell2mat(Success_cell.');
Success = Success_matrix(I_mix,:);
save Success.mat Success
