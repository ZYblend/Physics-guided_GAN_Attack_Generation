%% Run one time-domain simulation (with attack detection) to compare resiliency of four attack detection scenarios:
%    1. MLP1 (trained by random attacks)
%    2. MLP1 + pruning
%    3. MLP2 (trained by random attacks and automated generated attacks)
%    4. MLP2 + pruning
%
% Attacks used in this simulations: 2 attacks
%                            
% Evaluation of resiliency:
%                          (yc_deviation: deviation ratio of critical measurement from nominal critical measurement)
%                          (r_deviation: deviation ratio of detection residual from nominal detection resildual)
%                          (deviation ratio = 100*(attacked-nominal)/nominal)
%
% Yu Zheng, RASLab, FAMU-FSU College of Engineering, Tallahassee, 2021, Aug.


clear all
clc

% %% load attacks
% Attack = load('attack_for_time_domain_sim.mat').attack_for_time_domain_sim;

%% load system
Run_model_with_attackDetection

%% sample-based moving-horizon FDIA
% load specific attack dataset
datapath = append(motherpath,'\1_Attack_generation\Single_attack_data\');
attack = readmatrix(append(datapath,'attacks.csv'));
I_attack = readmatrix(append(datapath,'I_attack.csv'));
y1_effect = readmatrix(append(datapath,'y1_effect.csv'));
y2_detect = readmatrix(append(datapath,'y2_detect.csv'));

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
        
        
% %% Get results
% time_vec  = simOut.logsout.getElement('r_nominal').Values.Time;
% 
% % State vectors
% % attack-free
% r_nominal = simOut.logsout.getElement('r_nominal').Values.Data; 
% yc_nominal = simOut.logsout.getElement('yc_nominal').Values.Data; 
% r_delta0 = r_nominal-r_nominal;
% yc_delta0 = (yc_nominal - yc_d)/yc_d;
% 
% x_hat_free = simOut.logsout.getElement('x_hat').Values.Data; 
% x = simOut.logsout.getElement('x').Values.Data;
% 
% % MLP1 + pruning
% r_attacked = simOut.logsout.getElement('r_attacked').Values.Data; 
% yc_attacked  = simOut.logsout.getElement('yc_attacked').Values.Data;
% r_delta = r_attacked-r_nominal;
% yc_delta = (yc_attacked - yc_d)/yc_d;
% 
% x_hat_attacked1 = simOut.logsout.getElement('x_hat_attacked').Values.Data; 
% 
% 
% % MLP1
% r_attacked2 = simOut.logsout.getElement('r_attacked2').Values.Data; 
% yc_attacked2  = simOut.logsout.getElement('yc_attacked2').Values.Data; 
% r_delta2 = r_attacked2-r_nominal;
% yc_delta2 = (yc_attacked2 - yc_d)/yc_d;
% 
% x_hat_attacked2 = simOut.logsout.getElement('x_hat_attacked2').Values.Data; 
% 
% 
% % MLP2 + pruning
% r_attacked3 = simOut.logsout.getElement('r_attacked3').Values.Data; 
% yc_attacked3  = simOut.logsout.getElement('yc_attacked3').Values.Data;
% r_delta3 = r_attacked3-r_nominal;
% yc_delta3 = (yc_attacked3 - yc_d)/yc_d;
% 
% x_hat_attacked3 = simOut.logsout.getElement('x_hat_attacked3').Values.Data; 
% 
% % MLP2
% r_attacked4 = simOut.logsout.getElement('r_attacked4').Values.Data; 
% yc_attacked4  = simOut.logsout.getElement('yc_attacked4').Values.Data;
% r_delta4 = r_attacked4-r_nominal;
% yc_delta4 = (yc_attacked4 - yc_d)/yc_d;
% 
% x_hat_attacked4 = simOut.logsout.getElement('x_hat_attacked4').Values.Data; 
% 
% % calculate deviation ratio
% yc_max_delta = max([abs(yc_delta-yc_delta0);abs(yc_delta2-yc_delta0); abs(yc_delta3-yc_delta0); abs(yc_delta4-yc_delta0)]);
% yc_deviation = abs(yc_delta-yc_delta0)/yc_max_delta;
% yc_deviation2 = abs(yc_delta2-yc_delta0)/yc_max_delta;
% yc_deviation3 = abs(yc_delta3-yc_delta0)/yc_max_delta;
% yc_deviation4 = abs(yc_delta4-yc_delta0)/yc_max_delta;
% 
% r_max_delta = max([abs(r_delta-r_delta0);abs(r_delta2-r_delta0); abs(r_delta3-r_delta0); abs(r_delta4-r_delta0)]);
% r_deviation = abs(r_delta-r_delta0)/r_max_delta;
% r_deviation2 = abs(r_delta2-r_delta0)/r_max_delta;
% r_deviation3 = abs(r_delta3-r_delta0)/r_max_delta;
% r_deviation4 = abs(r_delta4-r_delta0)/r_max_delta;
% 
% % precision
% ppv1 = simOut.logsout.getElement('PPV1').Values.Data; 
% ppv1_eta = simOut.logsout.getElement('PPV1_eta').Values.Data; 
% 
% ppv2 = simOut.logsout.getElement('PPV2').Values.Data; 
% ppv2_eta = simOut.logsout.getElement('PPV2_eta').Values.Data; 
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
yc_deviation = 100*abs(yc_nominal-yc_attacked);
yc_deviation2 = 100*abs(yc_nominal-yc_attacked2);
yc_deviation3 = 100*abs(yc_nominal-yc_attacked3);
yc_deviation4 = 100*abs(yc_nominal-yc_attacked4);
yc_max = max(max([yc_deviation, yc_deviation2, yc_deviation3, yc_deviation4]));
yc_deviation = yc_deviation/yc_max;
yc_deviation2 = yc_deviation2/yc_max;
yc_deviation3 = yc_deviation3/yc_max;
yc_deviation4 = yc_deviation4/yc_max;


r_deviation = 100*abs(r_nominal-r_attacked);
r_deviation2 = 100*abs(r_nominal-r_attacked2);
r_deviation3 = 100*abs(r_nominal-r_attacked3);
r_deviation4 = 100*abs(r_nominal-r_attacked4);
r_max = max(max([r_deviation, r_deviation2, r_deviation3, r_deviation4]));
r_deviation = r_deviation/r_max;
r_deviation2 = r_deviation2/r_max;
r_deviation3 = r_deviation3/r_max;
r_deviation4 = r_deviation4/r_max;

% Precision
ppv1 = simOut.logsout.getElement('PPV1').Values.Data; 
ppv1_eta = simOut.logsout.getElement('PPV1_eta').Values.Data; 

ppv2 = simOut.logsout.getElement('PPV2').Values.Data; 
ppv2_eta = simOut.logsout.getElement('PPV2_eta').Values.Data;

r = {r_deviation2, r_deviation,r_deviation4,r_deviation3};
yc = {yc_deviation2,yc_deviation,yc_deviation4,yc_deviation3};

save r.mat r
save yc.mat yc
save time.mat time_vec


%% plotting
LW = 2;
FS = 15;

figure (1)
subplot(2,1,1)
plot(time_vec,r_deviation2,'g',time_vec,r_deviation,'b',time_vec,r_deviation4,'r',time_vec,r_deviation3,'k','LineWidth',LW)
legend('MLP1','MLP1+pruning','MLP2','MLP2+pruning');
ylabel('\Delta r (%)')
xlabel('Time instance')


ax = gca;
ax.LineWidth = LW-0.5;
ax.FontSize = FS;

subplot(2,1,2)
plot(time_vec,yc_deviation2,'g',time_vec,yc_deviation,'b',time_vec,yc_deviation4,'r',time_vec,yc_deviation3,'k','LineWidth',LW)
legend('MLP1','MLP1+pruning','MLP2','MLP2+pruning');
ylabel('\Delta y (%)')
xlabel('Time instance')


ax = gca;
ax.LineWidth = LW-0.5;
ax.FontSize = FS;

figure (2)
subplot(2,1,1)
plot(time_vec,r_deviation3,'k','LineWidth',LW)

ylabel('\Delta pr (%)')
xlabel('Time instance')

ax = gca;
ax.LineWidth = LW-0.5;
ax.FontSize = FS;

subplot(2,1,2)
plot(time_vec,yc_deviation3,'k','LineWidth',LW)

ylabel('\Delta py (%)')
xlabel('Time instance')

ax = gca;
ax.LineWidth = LW-0.5;
ax.FontSize = FS;

%% additional plot (estimation error compare)
x_hat_free = simOut.logsout.getElement('x_hat').Values.Data; 
x = simOut.logsout.getElement('x').Values.Data;
x_hat_attacked1 = simOut.logsout.getElement('x_hat_attacked').Values.Data; 
x_hat_attacked2 = simOut.logsout.getElement('x_hat_attacked2').Values.Data; 
x_hat_attacked3 = simOut.logsout.getElement('x_hat_attacked3').Values.Data; 
x_hat_attacked4 = simOut.logsout.getElement('x_hat_attacked4').Values.Data; 


normal_deviation = 0;
max_deriv = max(max([vecnorm(x - x_hat_free,2,2), vecnorm(x - x_hat_attacked1,2,2),vecnorm(x - x_hat_attacked2,2,2),  vecnorm(x - x_hat_attacked3,2,2), vecnorm(x - x_hat_attacked4,2,2)]));

deviation1 = vecnorm(x - x_hat_attacked1,2,2)/max_deriv;
deviation2 = vecnorm(x - x_hat_attacked2,2,2)/max_deriv;
deviation3 = vecnorm(x - x_hat_attacked3,2,2)/max_deriv;
deviation4 = vecnorm(x - x_hat_attacked4,2,2)/max_deriv;
deviation1(1:83) = 0;
deviation2(1:83) = 0;
deviation3(1:83) = 0;
deviation4(1:83) = 0;

figure
subplot(2,1,1)
plot(time_vec,deviation4,'r',time_vec,deviation3,'k',time_vec,deviation1,'b',time_vec, deviation2,'g','LineWidth',LW)
legend('MLP2','MLP2+pruning','MLP1+pruning','MLP1');
ylabel('\Delta x (%)');
xlabel('Time instance');
ylim([0 0.1])
subplot(2,1,2)
plot(time_vec,yc_deviation2,'g',time_vec,yc_deviation,'b',time_vec,yc_deviation4,'r',time_vec,yc_deviation3,'k','LineWidth',LW)
legend('MLP1','MLP1+pruning','MLP2','MLP2+pruning');
ylabel('\Delta y (%)')
xlabel('Time instance')





