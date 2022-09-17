%% Run simulation (with attack detection) to compare resiliency of four attack detection scenarios:
%    1. MLP1 (trained by random attacks)
%    2. MLP1 + pruning
%    3. MLP2 (trained by random attacks and automated generated attacks)
%    4. MLP2 + pruning
%
% Attacks used in simulations: (19*20 attack datasets)
%                             number of attacks from 2 to 20, for each, there are 20 dataset corresponding to different attack suppport
% Evaluation of resiliency:
%                          (yc_deviation: deviation ratio of critical measurement from nominal critical measurement)
%                          (r_deviation: deviation ratio of detection residual from nominal detection resildual)
%                          (deviation ratio = 100*(attacked-nominal)/nominal)
%                          1) ratio of safety: effect_ratio = 100%*(time_steps of yc_deviation < thresh1)/total time steps
%                          2) ratio of detection: detect_ratio = 100%*(time_steps of r_deviation > thresh2)/total time steps
%                         Better resiliency = bigger effect ratio + bigger detection ratio
%
% Yu Zheng, RASLab, FAMU-FSU College of Engineering, Tallahassee, 2021, Aug.


clear all
clc

%% load attacks
load Attacks.mat
iter_attacks = size(Attacks,1);                            % from n_attack=1 to n_attack=30 
iter_sum_per_attack = size(Attacks,2);                     % number of datasets for different n_attack
% iter_sum_per_attack = 3;
num_iter = iter_attacks*iter_sum_per_attack;               % total simulation iterations

%% load system
Run_model_with_attackDetection

%% simualtion
Effect_ratio_MLP1_pruning = zeros(iter_attacks,iter_sum_per_attack);
Effect_ratio_MLP1         = zeros(iter_attacks,iter_sum_per_attack);
Effect_ratio_MLP2_pruning = zeros(iter_attacks,iter_sum_per_attack);
Effect_ratio_MLP2         = zeros(iter_attacks,iter_sum_per_attack);
Detect_ratio_MLP1_pruning = zeros(iter_attacks,iter_sum_per_attack);
Detect_ratio_MLP1         = zeros(iter_attacks,iter_sum_per_attack);
Detect_ratio_MLP2_pruning = zeros(iter_attacks,iter_sum_per_attack);
Detect_ratio_MLP2         = zeros(iter_attacks,iter_sum_per_attack);

PPV1 = cell(iter_attacks,iter_sum_per_attack);
PPV1_eta = cell(iter_attacks,iter_sum_per_attack);
PPV2 = cell(iter_attacks,iter_sum_per_attack);
PPV2_eta = cell(iter_attacks,iter_sum_per_attack);

for iter1 = 1:iter_attacks
    for iter2 = 1:iter_sum_per_attack
        %% sample-based moving-horizon FDIA
        % load specific attack dataset
        I_attack = Attacks{iter1,iter2}{1,2};
        attack = Attacks{iter1,iter2}{1,1};
        y1_effect = Attacks{iter1,iter2}{1,3};
        y2_detect = Attacks{iter1,iter2}{1,4};
        
        n_attack = round(length(I_attack)/T);   % number of attacks
        I_attack = I_attack + ones(size(I_attack)); % notice: python index starts from 0
        I_attack_local = I_attack(1:n_attack);  % attacks upport for one time step

        % choose one best T attacks (y1_effect is biggest, y2_detect<tau2)
        % the T attacks will be injected at the beginning T time steps
        [~,I_best] = max(y1_effect(y2_detect<=tau2));
        attack_potential = attack(y2_detect<=tau2,:);
        attack_0 = attack_potential(I_best,:);

        % get attacks in row-wise (each row correponds to one attack)
        % atatck(m-by-n_attack*T) -- attack_tank (m*T -by- n_attack)
        attack_tank = zeros(size(attack,1)*T,n_attack);
        E0 = zeros(T,n_attack);
        for index = 1:T
            attack_tank((index-1)*size(attack,1)+1:index*size(attack,1),:) = attack(:,(index-1)*n_attack+1:index*n_attack);
            E0(index,:) = attack_0(:,(index-1)*n_attack+1:index*n_attack);
        end

        % get attacks in full dimension
        % attack_tank (m*T -by- n_attack) -- attack_tank_full (m*T-by-n_meas)
        attack_tank_full = zeros(size(attack_tank,1),n_meas);
        E0_full = zeros(T,n_meas);
        for iter = 1:n_attack
            index = I_attack_local(iter);
            attack_tank_full(:,index) = attack_tank(:,iter);
            E0_full(:,index) = E0(:,iter);
        end
        
        %% Run model
        simOut = sim('WDS_with_AttackDetection');
        
        
        %% Get results
        time_vec  = simOut.logsout.getElement('r_nominal').Values.Time;

        % State vectors
        % attack-free
        r_nominal = simOut.logsout.getElement('r_nominal').Values.Data; 
        yc_nominal = simOut.logsout.getElement('yc_nominal').Values.Data; 

        % MLP1 + pruning
        r_attacked = simOut.logsout.getElement('r_attacked').Values.Data; 
        yc_attacked  = simOut.logsout.getElement('yc_attacked').Values.Data;

        % MLP1
        r_attacked2 = simOut.logsout.getElement('r_attacked2').Values.Data; 
        yc_attacked2  = simOut.logsout.getElement('yc_attacked2').Values.Data; 

        % MLP2 + pruning
        r_attacked3 = simOut.logsout.getElement('r_attacked3').Values.Data; 
        yc_attacked3  = simOut.logsout.getElement('yc_attacked3').Values.Data;

        % MLP2
        r_attacked4 = simOut.logsout.getElement('r_attacked4').Values.Data; 
        yc_attacked4  = simOut.logsout.getElement('yc_attacked4').Values.Data;

        % calculate deviation ratio
        yc_deviation = 100*abs(yc_nominal-yc_attacked)./yc_nominal;
        yc_deviation2 = 100*abs(yc_nominal-yc_attacked2)./yc_nominal;
        yc_deviation3 = 100*abs(yc_nominal-yc_attacked3)./yc_nominal;
        yc_deviation4 = 100*abs(yc_nominal-yc_attacked4)./yc_nominal;

        r_deviation = 100*abs(r_nominal-r_attacked)./r_nominal;
        r_deviation2 = 100*abs(r_nominal-r_attacked2)./r_nominal;
        r_deviation3 = 100*abs(r_nominal-r_attacked3)./r_nominal;
        r_deviation4 = 100*abs(r_nominal-r_attacked4)./r_nominal;
        
        %% Evaluate Resiliency
        % calculate ratio of yc_deviation less than threshold
        Effect_ratio_MLP1_pruning(iter1,iter2) = sum(yc_deviation(end-N_attack+1:end)<=Thresh1)/N_attack;
        Effect_ratio_MLP1(iter1,iter2)         = sum(yc_deviation2(end-N_attack+1:end)<=Thresh1)/N_attack;
        Effect_ratio_MLP2_pruning(iter1,iter2) = sum(yc_deviation3(end-N_attack+1:end)<=Thresh1)/N_attack;
        Effect_ratio_MLP2(iter1,iter2)         = sum(yc_deviation4(end-N_attack+1:end)<=Thresh1)/N_attack;
        
        % calculate ratio of r_deviation bigger than threshold
        Detect_ratio_MLP1_pruning(iter1,iter2) = sum(r_deviation(end-N_attack+1:end)>=Thresh2)/N_attack;
        Detect_ratio_MLP1(iter1,iter2)         = sum(r_deviation2(end-N_attack+1:end)>=Thresh2)/N_attack;
        Detect_ratio_MLP2_pruning(iter1,iter2) = sum(r_deviation3(end-N_attack+1:end)>=Thresh2)/N_attack;
        Detect_ratio_MLP2(iter1,iter2)         = sum(r_deviation4(end-N_attack+1:end)>=Thresh2)/N_attack;
        
        % Precision
        ppv1 = simOut.logsout.getElement('PPV1').Values.Data; 
        ppv1_eta = simOut.logsout.getElement('PPV1_eta').Values.Data; 
        
        ppv2 = simOut.logsout.getElement('PPV2').Values.Data; 
        ppv2_eta = simOut.logsout.getElement('PPV2_eta').Values.Data;
        
        PPV1{iter1,iter2} = ppv1;
        PPV1_eta{iter1,iter2} = ppv1_eta;
        
        PPV2{iter1,iter2} = ppv2;
        PPV2_eta{iter1,iter2} = ppv2_eta;
        
        % end
        txt = ['simulation end at loop', num2str((iter1-1)*iter_sum_per_attack + iter2)];
        disp(txt)
    end
end
Sim_result_Effect = {Effect_ratio_MLP1_pruning, Effect_ratio_MLP1, Effect_ratio_MLP2_pruning, Effect_ratio_MLP2};
Sim_result_Detect = {Detect_ratio_MLP1_pruning, Detect_ratio_MLP1, Detect_ratio_MLP2_pruning, Detect_ratio_MLP2};
Sim_results = {Sim_result_Effect,Sim_result_Detect};
save Sim_results.mat Sim_results

PPV = {PPV1,PPV1_eta,PPV2,PPV2_eta};
save PPV.mat PPV



