%% Precision table
%
% Data structure:
%                PPV: cell array (1-by-4)
%                     - PPV{1,1}: PPV1 (precision of MLP1) {19-by-20)
%                     - PPV{1,2}: PPV1_pruning (precision of MLP1+pruning) {19-by-20)
%                     - PPV{1,3}: PPV2 (precision of MLP2) {19-by-20)
%                     - PPV{1,4}: PPV2_pruning (precision of MLP2+pruning) {19-by-20)
%                PPV1, PPV1_pruning, PPV2, PPV2_pruning:
%                     Row: number of attack (2-20)
%                     Column: different attack support
%
% Yu Zheng, RASLab, FAMU-FSU College of Engineering, Tallahassee, 2021, Aug.

%% unpack PPVs
load PPV.mat
PPV1 = PPV{1,1};
PPV1_pruning = PPV{1,2};
PPV2 = PPV{1,3};
PPV2_pruning = PPV{1,4};


%% get mean value for each number of attack and each attack support
[row_num, col_num] = size(PPV1);
PPV1_mean          = zeros(row_num,col_num);
PPV1_pruning_mean  = zeros(row_num,col_num);
PPV2_mean          = zeros(row_num,col_num);
PPV2_pruning_mean  = zeros(row_num,col_num);

parfor iter1 = 1:row_num
    for iter2 = 1:col_num
        PPV1_mean(iter1,iter2) = mean(PPV1{iter1,iter2});
        PPV1_pruning_mean(iter1,iter2) = mean(PPV1_pruning{iter1,iter2});
        PPV2_mean(iter1,iter2) = mean(PPV2{iter1,iter2});
        PPV2_pruning_mean(iter1,iter2) = mean(PPV2_pruning{iter1,iter2});
    end
end

%% Mean for each number of attacks
PPV1_mean_per_num         = mean(PPV1_mean,2);
PPV1_pruning_mean_per_num = mean(PPV1_pruning_mean,2);
PPV2_mean_per_num         = mean(PPV2_mean,2);
PPV2_pruning_mean_per_num = mean(PPV2_pruning_mean,2);

PPV_mean_per_num = [PPV1_mean_per_num, PPV1_pruning_mean_per_num, PPV2_mean_per_num, PPV2_pruning_mean_per_num];


%% Max for each number of attacks
PPV1_max_per_num         = max(PPV1_mean,[],2);
PPV1_pruning_max_per_num = max(PPV1_pruning_mean,[],2);
PPV2_max_per_num         = max(PPV2_mean,[],2);
PPV2_pruning_max_per_num = max(PPV2_pruning_mean,[],2);

PPV_max_per_num = [PPV1_max_per_num, PPV1_pruning_max_per_num, PPV2_max_per_num, PPV2_pruning_max_per_num];

%% Min for each number of attacks
PPV1_min_per_num         = min(PPV1_mean,[],2);
PPV1_pruning_min_per_num = min(PPV1_pruning_mean,[],2);
PPV2_min_per_num         = min(PPV2_mean,[],2);
PPV2_pruning_min_per_num = min(PPV2_pruning_mean,[],2);

PPV_min_per_num = [PPV1_min_per_num, PPV1_pruning_min_per_num, PPV2_min_per_num, PPV2_pruning_min_per_num];

%% plot got mean precision
LW = 2;
FS = 12;

num_attack = linspace(1,30,30);
figure (1)
plot(num_attack,PPV1_mean_per_num,'g','LineWidth',LW);
hold on, plot(num_attack,PPV1_pruning_mean_per_num,'b','LineWidth',LW);
hold on, plot(num_attack,PPV2_mean_per_num,'r','LineWidth',LW);
hold on, plot(num_attack,PPV2_pruning_mean_per_num,'k','LineWidth',LW);
legend('MLP1','MLP1+pruning','MLP2','MLP2+pruning');
ylabel('Precision (PPV)')
xlabel('Num of attacks')

% xlim([2 20]);

ax = gca;
ax.LineWidth = LW-0.5;
ax.FontSize = FS;
