%% Plot for final results
clear all
clc
%% load data
load  Sim_results.mat

Sim_result_Effect = Sim_results{1,1};
Sim_result_Detect = Sim_results{1,2};

Effect_ratio_MLP1_pruning = Sim_result_Effect{1,1};
Effect_ratio_MLP1         = Sim_result_Effect{1,2};
Effect_ratio_MLP2_pruning = Sim_result_Effect{1,3};
Effect_ratio_MLP2         = Sim_result_Effect{1,4};

Detect_ratio_MLP1_pruning = Sim_result_Detect{1,1};
Detect_ratio_MLP1         = Sim_result_Detect{1,2};
Detect_ratio_MLP2_pruning = Sim_result_Detect{1,3};
Detect_ratio_MLP2         = Sim_result_Detect{1,4};

% Mean
Effect_ratio_MLP1_pruning_mean = mean(Effect_ratio_MLP1_pruning,2);
Effect_ratio_MLP1_mean         = mean(Effect_ratio_MLP1,2);
Effect_ratio_MLP2_pruning_mean = mean(Effect_ratio_MLP2_pruning,2);
Effect_ratio_MLP2_mean         = mean(Effect_ratio_MLP2,2);

Detect_ratio_MLP1_pruning_mean = mean(Detect_ratio_MLP1_pruning,2);
Detect_ratio_MLP1_mean         = mean(Detect_ratio_MLP1,2);
Detect_ratio_MLP2_pruning_mean = mean(Detect_ratio_MLP2_pruning,2);
Detect_ratio_MLP2_mean         = mean(Detect_ratio_MLP2,2);

% Max
Effect_ratio_MLP1_pruning_max = max(Effect_ratio_MLP1_pruning,[],2);
Effect_ratio_MLP1_max         = max(Effect_ratio_MLP1,[],2);
Effect_ratio_MLP2_pruning_max = max(Effect_ratio_MLP2_pruning,[],2);
Effect_ratio_MLP2_max         = max(Effect_ratio_MLP2,[],2);

Detect_ratio_MLP1_pruning_max = max(Detect_ratio_MLP1_pruning,[],2);
Detect_ratio_MLP1_max         = max(Detect_ratio_MLP1,[],2);
Detect_ratio_MLP2_pruning_max = max(Detect_ratio_MLP2_pruning,[],2);
Detect_ratio_MLP2_max         = max(Detect_ratio_MLP2,[],2);

% Min
Effect_ratio_MLP1_pruning_min = min(Effect_ratio_MLP1_pruning,[],2);
Effect_ratio_MLP1_min         = min(Effect_ratio_MLP1,[],2);
Effect_ratio_MLP2_pruning_min = min(Effect_ratio_MLP2_pruning,[],2);
Effect_ratio_MLP2_min         = min(Effect_ratio_MLP2,[],2);

Detect_ratio_MLP1_pruning_min = min(Detect_ratio_MLP1_pruning,[],2);
Detect_ratio_MLP1_min         = min(Detect_ratio_MLP1,[],2);
Detect_ratio_MLP2_pruning_min = min(Detect_ratio_MLP2_pruning,[],2);
Detect_ratio_MLP2_min         = min(Detect_ratio_MLP2,[],2);

%% plotting
num_attacks = linspace(1,30,30);
figure (1)
subplot(1,2,1)
plot(num_attacks,Effect_ratio_MLP1_pruning_mean);
hold on, plot(num_attacks,Effect_ratio_MLP1_mean);
hold on, plot(num_attacks,Effect_ratio_MLP2_pruning_mean);
hold on, plot(num_attacks,Effect_ratio_MLP2_mean);
xlabel('Attack Percentage');
ylabel('Ratio of safty');
legend('MLP1+pruning', 'MLP1', 'MLP2+pruning', 'MLP2');
% legend('MLP2',  'MLP2+pruning');

subplot(1,2,2)
plot(num_attacks,Detect_ratio_MLP1_pruning_mean);
hold on, plot(num_attacks,Detect_ratio_MLP1_mean);
hold on, plot(num_attacks,Detect_ratio_MLP2_pruning_mean);
hold on, plot(num_attacks,Detect_ratio_MLP2_mean);
xlabel('Attack Percentage');
ylabel('Ratio of Detection');
legend('MLP1+pruning', 'MLP1', 'MLP2+pruning', 'MLP2');
% legend('MLP2',  'MLP2+pruning');

%% Box plot (medium value)
figure (2)
subplot(4,2,1)
boxplot(Effect_ratio_MLP1.');
ylim([0 1])
subplot(4,2,3)
boxplot(Effect_ratio_MLP1_pruning.');
ylim([0 1])
subplot(4,2,5)
boxplot(Effect_ratio_MLP2.');
ylim([0 1])
subplot(4,2,7)
boxplot(Effect_ratio_MLP2_pruning.');
ylim([0 1])

subplot(4,2,2)
boxplot(Detect_ratio_MLP1.');
subplot(4,2,4)
boxplot(Detect_ratio_MLP1_pruning.');
subplot(4,2,6)
boxplot(Detect_ratio_MLP2.');
subplot(4,2,8)
boxplot(Detect_ratio_MLP2_pruning.');


%% plot mean with standard deviation
LW = 2;
FS = 15;

% effect
figure (3)
num_attacks = linspace(1,30,30);
num_attacks2 = [num_attacks, fliplr(num_attacks)];

%%%%%%%%%%%%%%%%%
subplot(4,2,3)
plot(num_attacks,Effect_ratio_MLP1_pruning_mean,'b','LineWidth',LW);

inbetween_MLP1_pruning = [Effect_ratio_MLP1_pruning_max.', fliplr(Effect_ratio_MLP1_pruning_min.')];
hold on, h11 = fill(num_attacks2,inbetween_MLP1_pruning,'b');
set(h11,'facealpha',.3)

% xlabel('Num of Attack');
ylabel('pSNO');
legend('MLP1+pruning','');
% xlim([2,20]);
ylim([0 1]);
ax = gca;
ax.LineWidth = LW-0.5;
ax.FontSize = FS;

%%%%%%%%%%%%%%%%%%%%%%
subplot(4,2,1)
plot(num_attacks,Effect_ratio_MLP1_mean,'g','LineWidth',LW);

inbetween_MLP1 = [Effect_ratio_MLP1_max.', fliplr(Effect_ratio_MLP1_min.')];
hold on, h12 = fill(num_attacks2,inbetween_MLP1,'g');
set(h12,'facealpha',.3)

% xlabel('Num of Attack');
ylabel('pSNO');
legend('MLP1','');
% xlim([2,20]);
ylim([0 1]);

ax = gca;
ax.LineWidth = LW-0.5;
ax.FontSize = FS;

%%%%%%%%%%%%%%%%%%%%
subplot(4,2,7)
plot(num_attacks,Effect_ratio_MLP2_pruning_mean,'k','LineWidth',LW);

inbetween_MLP2_pruning = [Effect_ratio_MLP2_pruning_max.', fliplr(Effect_ratio_MLP2_pruning_min.')];
hold on, h13 = fill(num_attacks2,inbetween_MLP2_pruning,'k');
set(h13,'facealpha',.3)

xlabel('Num of Attack');
ylabel('pSNO');
legend('MLP2+pruning','');
% xlim([2,20]);

ax = gca;
ax.LineWidth = LW-0.5;
ax.FontSize = FS;

%%%%%%%%%%%%%%%%%%%%
subplot(4,2,5)
plot(num_attacks,Effect_ratio_MLP2_mean,'r','LineWidth',LW);

inbetween_MLP2 = [Effect_ratio_MLP2_max.', fliplr(Effect_ratio_MLP2_min.')];
hold on, h13 = fill(num_attacks2,inbetween_MLP2,'r');
set(h13,'facealpha',.3)

% xlabel('Num of Attack');
ylabel('pSNO');
legend('MLP2','');
% xlim([2,20]);

ax = gca;
ax.LineWidth = LW-0.5;
ax.FontSize = FS;

% Detect
%%%%%%%%%%%%%%%%%%%
subplot(4,2,4)
plot(num_attacks,Detect_ratio_MLP1_pruning_mean,'b','LineWidth',LW);

inbetween_MLP1_pruning = [Detect_ratio_MLP1_pruning_max.', fliplr(Detect_ratio_MLP1_pruning_min.')];
hold on, h11 = fill(num_attacks2,inbetween_MLP1_pruning,'b');
set(h11,'facealpha',.3)

xlabel('Num of Attack');
ylabel('pTA');
legend('MLP1+pruning','');
ylim([0 0.3])
% xlim([2,20]);

ax = gca;
ax.LineWidth = LW-0.5;
ax.FontSize = FS;

%%%%%%%%%%%%%%%%%%
subplot(4,2,2)
plot(num_attacks,Detect_ratio_MLP1_mean,'g','LineWidth',LW);

inbetween_MLP1 = [Detect_ratio_MLP1_max.', fliplr(Detect_ratio_MLP1_min.')];
hold on, h12 = fill(num_attacks2,inbetween_MLP1,'g');
set(h12,'facealpha',.3)

% xlabel('Num of Attack');
ylabel('pTA');
legend('MLP1','');
ylim([0 0.3])
% xlim([2,20]);

ax = gca;
ax.LineWidth = LW-0.5;
ax.FontSize = FS;

%%%%%%%%%%%%%%%%%%%%%%%
subplot(4,2,8)
plot(num_attacks,Detect_ratio_MLP2_pruning_mean,'k','LineWidth',LW);

inbetween_MLP2_pruning = [Detect_ratio_MLP2_pruning_max.', fliplr(Detect_ratio_MLP2_pruning_min.')];
hold on, h13 = fill(num_attacks2,inbetween_MLP2_pruning,'k');
set(h13,'facealpha',.3)

xlabel('Num of Attack');
ylabel('pTA');
legend('MLP2+pruning','');
ylim([0 0.3])
% xlim([2,20]);

ax = gca;
ax.LineWidth = LW-0.5;
ax.FontSize = FS;

%%%%%%%%%%%%%%%%%%%%%%%
subplot(4,2,6)
plot(num_attacks,Detect_ratio_MLP2_mean,'r','LineWidth',LW);

inbetween_MLP2 = [Detect_ratio_MLP2_max.', fliplr(Detect_ratio_MLP2_min.')];
hold on, h13 = fill(num_attacks2,inbetween_MLP2,'r');
set(h13,'facealpha',.3)

% xlabel('Num of Attack');
ylabel('pTA');
legend('MLP2','');
ylim([0 0.3])
% xlim([2,20]);

ax = gca;
ax.LineWidth = LW-0.5;
ax.FontSize = FS;


