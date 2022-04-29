%% Run model for simulation (without attack detection and localization)
% Description: This file is to prepare simulation parameters (to generate training dataset for attack detection algorithm)
%              simulink files contain two simulation side by side: 1) nominal simulation 2)attacked simulation
% Content:
%         1. load system matrices
%         2. Check observability and controllabiltiy
%         3. Get T-horizon system matrices
%         4. Get observer gains for L2 observer
%         5. Pole placement for gain Control design
%         6. Simulation Initialization
%         7. Runing parameters (time frame...)
%         8. parameters for sample-based moving-horizon FDIA
%
% Yu Zheng, RASLab, FAMU-FSU College of Engineering, Tallahassee, 2021, Aug.



%% 1-7 Run model
Run_model

%% 8. sample-based moving-horizon FDIA
% load data
n_attack = round(length(I_attack)/T);   % number of attacks
I_attack = I_attack + ones(size(I_attack)); % notice: python index starts from 0
I_attack_local = I_attack(1:n_attack);  % attacks upport for one time step
                                                         
M1 = readmatrix('M1.csv');
M2 = readmatrix('M2.csv');

tau1 = readmatrix('tau1.csv');
tau2 = readmatrix('tau2.csv');

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
