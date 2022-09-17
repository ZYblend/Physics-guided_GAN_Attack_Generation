%% This file is to align attack dataset in accending order of number of attacks
clear all
clc

% add path
currentpath = pwd;
motherpath = erase(currentpath,"\3_Sim_with_detector");
addpath(append(motherpath,'\1_Attack_generation'));


%% load attacks
All_attacks = load('All_attacks.mat').MyData;  % load attack dataset

iter_attacks = 30;                            % from n_attack=1 to n_attack=30 
iter_sum_per_attack = round(size(All_attacks,2)/iter_attacks);   % number of datasets for different n_attack
num_iter = size(All_attacks,2);               % total simulation iterations

num_attacks_seq = linspace(1,30,30);

T = 8;

Attacks = cell(iter_attacks,iter_sum_per_attack);
idx = zeros(iter_attacks,1);
for iter = 1:num_iter

    % load specific attack dataset
    I_attack  = All_attacks{1,iter}{1,2};
%     attack    = All_attacks{1,iter}{1,1};
%     y1_effect = All_attacks{1,iter}{1,3};
%     y2_detect = All_attacks{1,iter}{1,4};
    
    n_attack = round(length(I_attack)/T);   % number of attacks
    
    for iter2 = 1:iter_attacks
        if n_attack == num_attacks_seq(iter2)
            idx(iter2) = idx(iter2)+1;
            Attacks{n_attack,idx(iter2)} = All_attacks{1,iter};
        end
    end
end

save('Attacks.mat', 'Attacks', '-v7.3')

