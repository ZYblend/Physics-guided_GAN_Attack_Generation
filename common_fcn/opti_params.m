function [H0,H1,F] = opti_params(A,B,C,T)
% [H0,H1,F] = opti_params(A,B,C,T)
% For receding L2 Observre
%
% Input:
%   - A [n-by-n]          : System dynamic matrix
%   - B [n-by-l]          : System dynamic input matrix
%   - C [m-by-n]          : System dynamic output matrix
%   - T [Scalar (T>=n)]   : Observer receding horizon
%   
% Output:
%   - H0 [m*T-by-n]       : observer state-ouput linear map 
%                              (Phi__T is the last m-rows of Phi_T)
%   - H1 [m*T-by-l*T]     : observer input-output linear map
%                               (H__T is the last m-rows of H_T)
%   - F [n-by-l*T]        : Observer input-state propagation matrix
%                              Used for propagating the argmin value to the
%                              estimated state at the current instant
%
% Yu Zheng, RASLab, FAMU-FSU College of Engineering, Tallahassee, 2021, Aug.

%% Extracting relevant sizes
[m,n] = size(C);   % m = number of measured outputs, n = number of states
l     = size(B,2); % number of inputs


%% Calculating H0

% initializing
H0   = zeros(m*T,n);  

% building the body of H0
for iter = 1:T
    H0(m*(iter-1)+1:m*iter,:) = C*mpower(A,iter);
end


%% Calculating H1

% initializing
H1 = zeros(m*T,l*T);

% building the body of H1
for iter_1 = 1:T-1
    for iter_2 = 1:iter_1
        row_indices = m*(iter_1-1)+1:m*iter_1;
        col_indices = l*(iter_2-1)+1:l*(iter_2);
        
        H1(row_indices,col_indices) = C*mpower(A,iter_1-iter_2)*B;
    end
end


%% Calculating F

% initialization
F = zeros(n,l*T);

% building the body of F
for iter = 1:T
    F(:,(iter-1)*l+1:iter*l) = mpower(A,T-iter)*B;
end


end