function [perct_safe_nodes,k_bounds] = getRedundancyBound(p,n_samples,n_points)
% [perct_safe_nodes,k_bounds] = getRedundancyBound(p,n_samples,n_points)
%   calculates the expected redundancy bounds 
%       sum (e^-i r_i), i = 0:|T^c|
%   as a function of percentage number of safe nodes.
%   Algorithm:
%      For each |T^c|, 
%        a. Generate n_samples random T^c
%             -> For each sample, calculate the redundancy bound
%        b. Calculate the mean value of the redundancy bound
%
% Inputs:
%  - p [m-by-1 or 1-by-m]: Vector of true rates for m channels
%  - n_samples [scalar]  : Integer number of samples to generate
%  - n_points [scalar]   : Integer number of points to use for |T^c|
%
% Outputs:
%  - perct_safe_nodes [n-by-1]: Vector of percentage safe nodes
%  - k_bounds         [n-by-1]: Vector of redundancy bounds

% Olugbenga Moses Anubi, 9/2/2021


%% Initialization and size extraction
n_nodes = length(p(:)); % The total numbe of nodes

frac_safe_nodes_min = 0.1; % minimum fraction of the number of safe nodes to consider
perct_safe_nodes     = 100*linspace(frac_safe_nodes_min,1,n_points).'; 

k_bounds             = zeros(n_points,1);

%% Code body
parfor index = 1:n_points
    % Generate n_samples random permutation for T^c
    % And calculate mean value of redundancy bounds over the generated samples
    n_safe_nodes_index = round(perct_safe_nodes(index)*n_nodes/100);
    k_bounds(index) = 0;
    for i_sample = 1:n_samples
        safe_support_i = randperm(n_nodes,n_safe_nodes_index);
        
        r_bar_i = pmf_PB(p(safe_support_i)); %#ok<PFBNS>
        k_bounds(index) = k_bounds(index) + (exp(-(0:n_safe_nodes_index))*r_bar_i)/n_samples; % k_bound+=sum(e^-i * r_i)/n_samples
    end
    
end
