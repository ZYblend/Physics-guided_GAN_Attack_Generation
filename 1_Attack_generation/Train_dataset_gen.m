%% This file is Entrance file for generating training dataset
%  Content:
%          1. load dataset (from pyhton): All_attacks.mat (MyData)
%                                           1) MyData{1,iter}{1,1}: attack data
%                                           2) MyData{1,iter}{1,2}: attack support (notice: python index starts from 0)
%                                           3) MyData{1,iter}{1,3}: effectiveness calculated by M1
%                                           4) MyData{1,iter}{1,4}: T-horizon cumulative detection power calculated by M2
%          2. Run "Get_results.m"
%          3. Mix dataset and finalize the final dataset
%              
% Yu Zheng, RASLab, FAMU-FSU College of Engineering, Tallahassee, 2021, Aug.

clear variables
close all
clc

% add path
currentpath = pwd;
motherpath = erase(currentpath,"\1_Attack_generation");
addpath(append(motherpath,'\Common_fcns'));

%% Load all attack datasets
load All_attacks.mat   % MyData

num_attack_dataset = size(MyData,2);  % number of attack datasets we have
% num_attack_dataset = 1;

% thresholds for deviation ratio in real system simulation
Tau1_real = 2;  % for effect
Tau2_real = 1.5;  % for detection

%% Run simulation and prepare training dataset
training_data = [];
for iter = 1:num_attack_dataset
    % load specific attack dataset
    I_attack = MyData{1,iter}{1,2};
    attack = MyData{1,iter}{1,1};
    y1_effect = MyData{1,iter}{1,3};
    y2_detect = MyData{1,iter}{1,4};
    
    Get_results   % Use script 'Get_results'
    data = [train_data{1}, train_data{2}];    % put the input and output in same row in order to pair them while mixing
    training_data{end+1} = data;
end

%% Mix dataset and finalize the final dataset
Train_data = cell2mat(training_data');                    % put training dataset in one matrix
Train_data = Train_data(randperm(size(Train_data,1)),:);  % mix
save Train_data.mat Train_data                 % save    
    
    
    
    
    
    