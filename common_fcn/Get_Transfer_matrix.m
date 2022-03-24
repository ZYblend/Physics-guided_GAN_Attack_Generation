function [M1,M2] = Get_Transfer_matrix(A,B,Cc,H0,K,T,n_meas)
%% [M1,M2] = Get_Transfer_matrix1(A,B,Cm,H0,K,T,n_meas)
% Get Transfer Mtraix M1, M2
% M1: transfer matrix from attack to T-horizon cumulative detection residual
%       M1 = Cc*B_bar*A.'*pinv(H0), where B_bar = BK
% M2: transfer matrix from attack to next-time-step critical measurement
%       M2 =(I-H0*pinv(H0))
% 
% Inputs:
%        - A [n-by-n],B [n-by-p]: state-state matrix, input-state matrix
%        - Cc:     [mc-by-n] critical measurement matrix
%        - T:      [scalar] time horizon
%        - H0 = [CA; CA^2; ... ; CA^T]: T-horizon measurement matrix, where C [m-by-n] is measurement matrix for one time step
%        - K:      [p-by-n] control gain
%        - n_meas: [scalar] measurement dimension
% Please refer to equation (14) in paper <Algorithm Design for Resilient Cyber-Physical Systems using Automated Attack Generative Models> 
%
% Yu Zheng, RASLab, FAMU-FSU College of Engineering, Tallahassee, 2021, Aug.

%% M1 & M2
H0_pinv = pinv(H0,0.001);
M2 = eye(T*n_meas,T*n_meas) - H0*H0_pinv;

B_bar = B*K;
M1 = Cc*B_bar*A.'*H0_pinv;

end
