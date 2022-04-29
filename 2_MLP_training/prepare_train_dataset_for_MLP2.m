%% Combine training data for MLP2
%  MLP1: trained by random attacks
%  MLP2: trained by random attacks and automated generated attacks
%
% Yu Zheng, RASLab, FAMU-FSU College of Engineering, Tallahassee, 2021, Aug.
clear all
clc

% add path
currentpath = pwd;
motherpath = erase(currentpath,"\2_MLP_training");
addpath(append(motherpath,'\1_Attack_generation'));

%% Load dataset
automated_attack_dataset = load('Train_data.mat').Train_data;
random_attack_dataset = load('random_dataset\Random_train_data.mat').Train_data_mixed;

%% get dataset for MLP1 and MLP2
Dataset_for_MLP1 = random_attack_dataset;
Dataset_for_MLP2 = [automated_attack_dataset; random_attack_dataset];

% mix Dataset_for_MLP2
Dataset_for_MLP2 = Dataset_for_MLP2(randperm(size(Dataset_for_MLP2,1)),:);  % mix

% save dataset
writematrix(Dataset_for_MLP1,'Dataset_for_MLP1.csv')
writematrix(Dataset_for_MLP2,'Dataset_for_MLP2.csv')

