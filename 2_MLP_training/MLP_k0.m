%% this file is to calculate the k0 for MLP attack detection algorithm
% k0: the lower-bound size of |S| for which (A,C_{S}) is observable
%
clear all
clc

% add path
currentpath = pwd;
motherpath = erase(currentpath,"\2_MLP_training");
addpath(append(motherpath,'\1_Attack_generation'));
addpath(append(motherpath,'\Common_fcns'));

%% Get MLP result (calculate confidence vector from ROC)
%%%%%%%%%%%%% 1. load MLPs %%%%%%%%%%%%%
% MLP1
w1_MLP1 = readmatrix('MLP1/w1.csv');  % [input_size,hidenlayer1_size]
w2_MLP1 = readmatrix('MLP1/w2.csv');  % [hidenlayer1_size,hidenlayer2_size]
w3_MLP1 = readmatrix('MLP1/w3.csv');  % [hidenlayer2_size,hidenlayer3_size]
w4_MLP1 = readmatrix('MLP1/w4.csv');  % [hidenlayer3_size,Output_size]
b1_MLP1 = readmatrix('MLP1/b1.csv');  % [hidenlayer1_size,1]
b2_MLP1 = readmatrix('MLP1/b2.csv');  % [hidenlayer2_size,1]
b3_MLP1 = readmatrix('MLP1/b3.csv');  % [hidenlayer3_size,1]
b4_MLP1 = readmatrix('MLP1/b4.csv');  % [Output_size,1]

% MLP2 (With automated attack generation)
w1_MLP2 = readmatrix('MLP2/w1.csv');  % [input_size,hidenlayer1_size]
w2_MLP2 = readmatrix('MLP2/w2.csv');  % [hidenlayer1_size,hidenlayer2_size]
w3_MLP2 = readmatrix('MLP2/w3.csv');  % [hidenlayer2_size,hidenlayer3_size]
w4_MLP2 = readmatrix('MLP2/w4.csv');  % [hidenlayer3_size,Output_size]
b1_MLP2 = readmatrix('MLP2/b1.csv');  % [hidenlayer1_size,1]
b2_MLP2 = readmatrix('MLP2/b2.csv');  % [hidenlayer2_size,1]
b3_MLP2 = readmatrix('MLP2/b3.csv');  % [hidenlayer3_size,1]
b4_MLP2 = readmatrix('MLP2/b4.csv');  % [Output_size,1]

%%%%%%%%%%%%% 2. laod test dataset %%%%%%%%%%%%%
Test_data = load('Train_data.mat').Train_data;  % [N_data,input_size+output_size]

n_test = round(0.9*length(Test_data));  
test_data = Test_data(randperm(size(Test_data,1),n_test),:);  % randomly choose 90% examples for testing

input_size = size(w1_MLP1,1);
test_input = test_data(:,1:input_size).';
test_output = test_data(:,input_size+1:end).';

%%%%%%%%%%%%% 3. Prediction %%%%%%%%%%%%%
% output of neural network (probability)
MLP1_pred_prob = execute_mlp_4layers(test_input,w1_MLP1,w2_MLP1,w3_MLP1,w4_MLP1,b1_MLP1,b2_MLP1,b3_MLP1,b4_MLP1); 
MLP2_pred_prob = execute_mlp_4layers(test_input,w1_MLP2,w2_MLP2,w3_MLP2,w4_MLP2,b1_MLP2,b2_MLP2,b3_MLP2,b4_MLP2);

% output of classifier (0,1)
MLP1_pred = (MLP1_pred_prob >=0.5);
MLP2_pred = (MLP2_pred_prob >=0.5);

% confidence vector
% (TP+TN)/(TP+TN+FP+FN)
P1 = sum(MLP1_pred==test_output,2)/(n_test);
P2 = sum(MLP2_pred==test_output,2)/(n_test);
% 
save P1.mat P1
save P2.mat P2



%% calculate k0
% n_test = 50000;
k0_MLP1 = zeros(1,n_test);
k0_MLP2 = zeros(1,n_test);

test_data = Test_data(randperm(size(Test_data,1),n_test),:);  % randomly choose 50000 examples for testing
test_input = test_data(:,1:input_size).';
test_output = test_data(:,input_size+1:end).';

MLP1_pred_prob = execute_mlp_4layers(test_input,w1_MLP1,w2_MLP1,w3_MLP1,w4_MLP1,b1_MLP1,b2_MLP1,b3_MLP1,b4_MLP1); 
MLP2_pred_prob = execute_mlp_4layers(test_input,w1_MLP2,w2_MLP2,w3_MLP2,w4_MLP2,b1_MLP2,b2_MLP2,b3_MLP2,b4_MLP2);

% output of classifier (0,1)
MLP1_pred = (MLP1_pred_prob >=0.5);
MLP2_pred = (MLP2_pred_prob >=0.5);

parfor iter = 1:n_test
%     p1 = MLP1_pred_prob(:,iter);
%     p2 = MLP2_pred_prob(:,iter);
    pred1 = MLP1_pred(:,iter);    % 1:safe, 0: attack
    pred2 = MLP2_pred(:,iter);
    %%%%%%%%%%%%%%% 1. closed form expression of pdf %%%%%%%%%%%%%%%%%
    r_MLP1 = pmf_PB(P1(pred1>=0.5));  
    r_MLP2 = pmf_PB(P2(pred2>=0.5));  

    L_MLP1 = sum(pred1>=0.5);    % number of safe nodes
    L_MLP2 = sum(pred2>=0.5);

    %%%%%%%%%%%%%%% 2. lower bound for e^{-k0} %%%%%%%%%%%%%%%%%
    e_power_i1 = ones(L_MLP1+1,1);
    e_power_i2 = ones(L_MLP2+1,1);
    for idx1 = 1:L_MLP1+1
       e_power_i1(idx1) = exp(1-idx1);
    end
    for idx2 = 1:L_MLP2+1
       e_power_i2(idx2) = exp(1-idx2);
    end

    LB_MLP1 = sum(e_power_i1.*r_MLP1(1:L_MLP1+1));
    LB_MLP2 = sum(e_power_i2.*r_MLP2(1:L_MLP2+1));

    %%%%%%%%%%%%%%% 3. upper bound for k0 %%%%%%%%%%%%%%%%%
    k0_MLP1(iter) = - log(LB_MLP1);
    k0_MLP2(iter) = - log(LB_MLP2);
end

%% plotting
font_size = 15;
line_weights = 1.5;
Edges   = linspace(0,61,30);

yyaxis left
histogram(k0_MLP1,Edges,'Normalization','probability','LineWidth',line_weights)
ax = gca;
ax.YLim = [0 1];
ax.FontSize = font_size;
ax.LineWidth = line_weights;
ylabel('Frequency','Color','k')

yyaxis right
hold on, histogram(k0_MLP2,Edges,'Normalization','probability','LineWidth',line_weights)
xlabel('Observability requirement after pruning')
% xticklabels({'strict','','','','loose'})
xlabel('max(k_0)','Color','k')

grid on
ax = gca;
ax.YLim = [0 1];
ax.FontSize = font_size;
ax.LineWidth = line_weights;
    
hold on, xline(33,'--r','LineWidth',line_weights,'label','Observability line')
% xticks([5 10 15 20 27 30 35 40])

legend('MLP1','MLP2')





