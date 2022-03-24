function e_i = getCurrentAttack(e_history,M1,M2,tau,e_data, T)
% e_i = getCurrentAttack(e_history,M1,M2,tau,e_data)
%  returns the current attack vector e_i that solves the following
%  optimization problem or returns zeros if infeasible:
%         Maximize: ||M1*[e_history; e_i]||_2 
%         Subject to: ||M2*[e_history; e_i||_2 <= tau
%                      e_i \in e_data
%  where
%     - e_history [n*(T-1)- vector]: vector of attack history in the moving
%                            window [i-T+1:i-1]
%     - e_data [N_data-by-n]: matrix of pre-generated attack vectors
%     - M1 [n_m1-by-(T*n)]  : Attack impact operator
%     - M2 [n_m2-by-(T*n)]  : Detection residual operator
%     - tau [scalar]        : Detection threshold
%     - T   [scalar]        : time horizon
% Notice:
%        n is the number of measurements
%        e_history and e_data are full attacked measurements


% Olugbenga Moses Anubi 
% Yu zheng
% RASLab, FAMU-FSU College of Engineering, Tallahassee, 2021, Aug.

%% parameters
% size
[N_data, n_meas] = size(e_data);


% Divide M1, M2 corresponding to e_history and e_i
M11 = M1(:,1:(T-1)*n_meas);     % corresponding to e_history
M12 = M1(:,(T-1)*n_meas+1:end); % corresponding to e_i
M21 = M2(:,1:(T-1)*n_meas);     % corresponding to e_history
M22 = M2(:,(T-1)*n_meas+1:end); % corresponding to e_i


%% Check Constrains
% ||M2*[e_history; e_i||_2^2 = ||M21*e_history||_2^2 + 2*e_history.'*M21.'*M22*e_i + e_i.'*M22.'*M22*e_i
a2 = norm(M21*e_history)^2; 
b2 = 2*e_history.'*M21.'*M22;
H2 = M22.'*M22;

% get mask for ||M2*[e_history; e_i||_2^2 <= tau^2 in e_data (feasible attack set)
mask = (a2*ones(N_data,1)+e_data*b2.'+sum(e_data.*(e_data*H2.'),2) <= (tau^2)*ones(N_data,1));


%% Find the best attack from feasible attack set
% check if mask is void
if sum(mask)>=1 % if not void, e_i = argmax ||M1*[e_history; e_i]||_2
    e_feasible = e_data(mask,:);
    
    % ||M1*[e_history; e_i]||_2^2 = ||M11*e_history||_2^2 + 2*e_history.'*M11.'*M12*e_i + e_i.'*M12.'*M12*e_i
    b1 = 2*e_history.'*M11.'*M12;
    H1 = M12.'*M12;
    
    % object: e_i = argmax 2*e_history.'*M11.'*M12*e_i + e_i.'*M12.'*M12*e_i
    obj = e_feasible*b1.'+sum(e_feasible.*(e_feasible*H1.'),2);
    [~,I_max] = max(obj);
    e_i = e_feasible(I_max,:).';
    
else % if void, set attacks are zero at current time step
    e_i = zeros(n_meas,1);
end


