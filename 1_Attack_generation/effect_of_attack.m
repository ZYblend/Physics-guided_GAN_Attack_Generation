%% test the effectiveness and stealthiness of generated attack through system simulation
%
clear all
clc

% add path
currentpath = pwd;
motherpath = erase(currentpath,"\1_Attack_generation");
addpath(append(motherpath,'\Common_fcns'));

% read attack data
attack = readmatrix('Single_attack_data/attacks.csv');
I_attack = readmatrix('Single_attack_data/I_attack.csv');
y1_effect = readmatrix('Single_attack_data/y1_effect.csv');
y2_detect = readmatrix('Single_attack_data/y2_detect.csv');

% thresholds
Tau1_real = 2.2; 
Tau2_real = 1.5;

% run simulation
Get_results   % Use script 'Get_results'

data = [train_data{1}, train_data{2}];    % put the input and output in same row in order to pair them while mixing

% plotting
Plot_effect_of_attack
