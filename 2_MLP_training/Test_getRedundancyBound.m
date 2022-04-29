clear
clc
% load data
load('P1.mat')
load('P2.mat')

% calling getRedundancyBound
n_samples = 1000;
n_points  = min(100,round(0.9*length(P1)));

[perct_safe_nodes,k_bounds1] = getRedundancyBound(P1,n_samples,n_points);
[~,k_bounds2]                = getRedundancyBound(P2,n_samples,n_points);

figure,
plot(perct_safe_nodes,k_bounds1,perct_safe_nodes,k_bounds2)
legend('MLP1','MLP2','Location','Best')

figure,
plot(perct_safe_nodes,-log(k_bounds1),perct_safe_nodes,-log(k_bounds2))
legend('MLP1','MLP2','Location','Best')