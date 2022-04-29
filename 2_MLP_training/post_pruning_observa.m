%% this file is to calculate the k0 for MLP attack detection algorithm
% k0: the lower-bound size of |S| for which (A,C_{S}) is observable
%
clear all
clc

% add path
currentpath = pwd;
motherpath = erase(currentpath,"\2_MLP_training");
addpath(append(motherpath,'\Common_fcns'));

%% Get MLP confidence vector from ROC
load P1.mat
load P2.mat

%% probability of k_0>42 (50000 samples)
size_start = 40;
n_meas = length(P1);
tot = n_meas-size_start+1;
n_samples = 50000;
n_perm_temp = round(1.1*n_samples); % used to generate random permution with possible repitition
obsv_thresh = 42;

size_support = linspace(size_start,n_meas,tot).';

% run samples
k0_MLP1 = nan(tot,n_samples);
k0_MLP2 = nan(tot,n_samples);
    
prob_MLP1 = zeros(1,tot);
prob_MLP2 = zeros(1,tot);
exp_k0_MLP1 = zeros(1,tot);
exp_k0_MLP2 = zeros(1,tot);
parfor idx = 1:tot
    % Generating unique supports
    k = size_support(idx);
    n_support =  min(nchoosek(n_meas,k),n_perm_temp);
    supports_temp = zeros(n_support,k);
    for count = 1:n_support
        supports_temp(count,:) = randperm(n_meas,k);
    end
    supports_temp_unique = unique(supports_temp,'rows');
    if(size(supports_temp_unique,1)>n_samples)
        supports = supports_temp_unique(1:n_samples,:);
    else
        supports = supports_temp_unique;
    end
    n_samples_local = size(supports,1);
    
    % run samples
    k0_MLP1_local = nan(1,n_samples);
    k0_MLP2_local = nan(1,n_samples);
    for iter = 1:n_samples_local
      
        % get random support T_hat_c (estimated support of safe nodes)
        support = supports(iter,:); % sort(randperm(n_meas,size_support(idx))).';
        indicator = zeros(n_meas,1);
        indicator(support) = 1;
        
        %%%%%%%%%%%%%%% 1. closed form expression of pdf %%%%%%%%%%%%%%%%%
        r_MLP1 = pmf_PB(P1(indicator>=0.5));
        r_MLP2 = pmf_PB(P2(indicator>=0.5));
        
        L_MLP = length(support);    % number of safe nodes
        
        %%%%%%%%%%%%%%% 2. lower bound for e^{-k0} %%%%%%%%%%%%%%%%%
        e_power_i1 = ones(L_MLP+1,1);
        e_power_i2 = ones(L_MLP+1,1);
        for idx1 = 1:L_MLP+1
            e_power_i1(idx1) = exp(1-idx1);
        end
        for idx2 = 1:L_MLP+1
            e_power_i2(idx2) = exp(1-idx2);
        end
        
        LB_MLP1 = sum(e_power_i1.*r_MLP1(1:L_MLP+1));
        LB_MLP2 = sum(e_power_i2.*r_MLP2(1:L_MLP+1));
        
        %%%%%%%%%%%%%%% 3. upper bound for k0 %%%%%%%%%%%%%%%%%
        k0_MLP1_local(iter) = - log(LB_MLP1);
        k0_MLP2_local(iter) = - log(LB_MLP2);
    end
    k0_MLP1(idx,:) = k0_MLP1_local;
    k0_MLP2(idx,:) = k0_MLP2_local;
    
    prob_MLP1(idx) = sum(k0_MLP1_local>=obsv_thresh)/n_samples_local;
    prob_MLP2(idx) = sum(k0_MLP2_local>=obsv_thresh)/n_samples_local;
    
    exp_k0_MLP1(idx) = mean(k0_MLP1_local,'omitnan');
    exp_k0_MLP2(idx) = mean(k0_MLP2_local,'omitnan');
end


%% plotting
figure,
font_size = 15;
line_weights = 2;

subplot(2,2,1)
plot(size_support,prob_MLP1,'ko-','LineWidth',line_weights);
% ylabel('Pr(> k_0)')
title('MLP1')
ax = gca;
ax.FontSize = font_size;
ax.LineWidth = line_weights;

subplot(2,2,2)
plot(size_support,prob_MLP2,'ko-','LineWidth',line_weights);
title('MLP2')
ax = gca;
ax.FontSize = font_size;
ax.LineWidth = line_weights;

subplot(2,2,3)
plot(size_support,exp_k0_MLP1,'k','LineWidth',line_weights);
hold on, plot(size_support,min(k0_MLP1,[],2),'r.','LineWidth',line_weights);
hold on, plot(size_support,max(k0_MLP1,[],2),'g.','LineWidth',line_weights);
hold on, plot(size_support,obsv_thresh*ones(tot,1),'r--','LineWidth',line_weights);
% ylabel('Mean(ramained number of measurements)')
% xlabel('|\hat{\mathcal{T}}^c|)')
ax = gca;
ax.FontSize = font_size;
ax.LineWidth = line_weights;
ax.YLim = [15 60];
% yticks([15 20 25 30])
legend('mean','min','max')

subplot(2,2,4)
plot(size_support,exp_k0_MLP2,'k','LineWidth',line_weights);
hold on, plot(size_support,min(k0_MLP2,[],2),'r.','LineWidth',line_weights);
hold on, plot(size_support,max(k0_MLP2,[],2),'g.','LineWidth',line_weights);
hold on, plot(size_support,obsv_thresh*ones(tot,1),'r--','LineWidth',line_weights);
% xlabel('|\hat{\mathcal{T}}^c|)')
ax = gca;
ax.FontSize = font_size;
ax.LineWidth = line_weights;
ax.YLim = [15 60];
% yticks([15 20 25 30])
legend('mean','min','max')


