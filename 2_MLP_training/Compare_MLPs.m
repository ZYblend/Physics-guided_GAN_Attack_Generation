%% Compare MLPs
% 1. Get trained weights of MLP and execute MLP feedfoward function
% 2. Implement two MLPs on a same testing attack dataset, then calcaulate localization precision
%                  MLP1: trained by random attacks
%                  MLP2: trained by random attacks and automated generated attacks
% Precision is calculated by: PPV = True positive/(True positive + False positive)
% (refer to: receiver operating characteristic)
%
% Yu Zheng, RASLab, FAMU-FSU College of Engineering, Tallahassee, 2021, Aug.
clear all
clc
% add path
currentpath = pwd;
motherpath = erase(currentpath,"\2_MLP_training");
addpath(append(motherpath,'\1_Attack_generation'));
addpath(append(motherpath,'\Common_fcns'));

%% load MLPs
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


%% laod test dataset
Test_data = load('Train_data.mat').Train_data;  % [N_data,input_size+output_size]

n_test = round(0.9*length(Test_data)); 
test_data = Test_data(randperm(size(Test_data,1),n_test),:);  % randomly choose 50000 examples for testing

input_size = size(w1_MLP1,1);
test_input = test_data(:,1:input_size).';
test_output = test_data(:,input_size+1:end).';


%% Prediction
% output of neural network (probability)
MLP1_pred_prob = execute_mlp_4layers(test_input,w1_MLP1,w2_MLP1,w3_MLP1,w4_MLP1,b1_MLP1,b2_MLP1,b3_MLP1,b4_MLP1); 
MLP2_pred_prob = execute_mlp_4layers(test_input,w1_MLP2,w2_MLP2,w3_MLP2,w4_MLP2,b1_MLP2,b2_MLP2,b3_MLP2,b4_MLP2);

% output of classifier (0,1)
MLP1_pred = (MLP1_pred_prob >=0.5);
MLP2_pred = (MLP2_pred_prob >=0.5);




%% Evaluation
% precision
PPV_MLP1 = sum(test_output.*MLP1_pred,1)./sum(MLP1_pred,1);
PPV_MLP2 = sum(test_output.*MLP2_pred,1)./sum(MLP2_pred,1);

% Mean of precision
PPV_MLP1_mean = sum(PPV_MLP1)/size(test_output,2);
PPV_MLP2_mean = sum(PPV_MLP2)/size(test_output,2);

% save results
save results/PPV_MLP1.mat PPV_MLP1
save results/PPV_MLP2.mat PPV_MLP2

save results/PPV_MLP1_mean.mat PPV_MLP1_mean
save results/PPV_MLP2_mean.mat PPV_MLP2_mean


%% plotting (histogram plotting)
font_size = 20;
Edges   = linspace(0.4,1,30);
yyaxis left
histogram(PPV_MLP1,Edges,'Normalization','probability')
ax = gca;
ax.YLim = [0 0.9];
yyaxis right
hold on, histogram(PPV_MLP2,Edges,'Normalization','probability')
xlabel('Localization Precision')
legend('MLP1','MLP2')
grid on
ax = gca;
%ax.YLim = [0 0.12];



