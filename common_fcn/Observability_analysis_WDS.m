%% Observability evaluation of system
% for different number of measurements 

Run_model

num_meas = n_meas-1+1;
meas_seq = linspace(1,n_meas,num_meas);
num_sample = 50000;

obsv_i = zeros(num_meas,num_sample);
parfor idx = 1:num_meas
    for iter = 1:num_sample
        S = randperm(n_meas,meas_seq(idx));
        C_S = C_obsv_d(S,:);
        if rank(obsv(A_bar_d,C_S)) == n_states
            obsv_i(idx,iter) = 1;
        end
    end
end
prob_obsv = sum(obsv_i,2)/num_sample;


%% plotting
LW = 2;
FS = 15;

plot(meas_seq,100*prob_obsv,'k','LineWidth',LW);
ylabel('Percentage of observable (A,C_S) (%)')
xlabel('|S|')
ax = gca;
ax.LineWidth = LW-0.5;
ax.FontSize = FS;
    

