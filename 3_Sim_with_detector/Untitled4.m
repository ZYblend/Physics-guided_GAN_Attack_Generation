        %% Get results
        time_vec  = out.logsout.getElement('r_nominal').Values.Time;

        % State vectors
        % attack-free
        r_nominal = out.logsout.getElement('r_nominal').Values.Data; 
        yc_nominal = out.logsout.getElement('yc_nominal').Values.Data; 
        
        q_eta_hat = out.logsout.getElement('q_eta_hat').Values.Data; 

        % MLP1 + pruning
        r_attacked = out.logsout.getElement('r_attacked').Values.Data; 
        yc_attacked  = out.logsout.getElement('yc_attacked').Values.Data;

        % MLP1
        r_attacked2 = out.logsout.getElement('r_attacked2').Values.Data; 
        yc_attacked2  = out.logsout.getElement('yc_attacked2').Values.Data; 

        % MLP2 + pruning
        r_attacked3 = out.logsout.getElement('r_attacked3').Values.Data; 
        yc_attacked3  = out.logsout.getElement('yc_attacked3').Values.Data;

        % MLP2
        r_attacked4 = out.logsout.getElement('r_attacked4').Values.Data; 
        yc_attacked4  = out.logsout.getElement('yc_attacked4').Values.Data;

        % calculate deviation ratio
        yc_deviation = 100*abs(yc_nominal-yc_attacked)./yc_nominal;
        yc_deviation2 = 100*abs(yc_nominal-yc_attacked2)./yc_nominal;
        yc_deviation3 = 100*abs(yc_nominal-yc_attacked3)./yc_nominal;
        yc_deviation4 = 100*abs(yc_nominal-yc_attacked4)./yc_nominal;

        r_deviation = 100*abs(r_nominal-r_attacked)./r_nominal;
        r_deviation2 = 100*abs(r_nominal-r_attacked2)./r_nominal;
        r_deviation3 = 100*abs(r_nominal-r_attacked3)./r_nominal;
        r_deviation4 = 100*abs(r_nominal-r_attacked4)./r_nominal;
        
        %% Evaluate Resiliency
        % calculate ratio of yc_deviation less than threshold
        Effect_ratio_MLP1_pruning = sum(yc_deviation(end-N_attack+1:end)<=Thresh1)/N_attack;
        Effect_ratio_MLP1       = sum(yc_deviation2(end-N_attack+1:end)<=Thresh1)/N_attack;
        Effect_ratio_MLP2_pruning= sum(yc_deviation3(end-N_attack+1:end)<=Thresh1)/N_attack;
        Effect_ratio_MLP2       = sum(yc_deviation4(end-N_attack+1:end)<=Thresh1)/N_attack;
        
        % calculate ratio of r_deviation bigger than threshold
        Detect_ratio_MLP1_pruning = sum(r_deviation(end-N_attack+1:end)>=Thresh2)/N_attack;
        Detect_ratio_MLP1      = sum(r_deviation2(end-N_attack+1:end)>=Thresh2)/N_attack;
        Detect_ratio_MLP2_pruning = sum(r_deviation3(end-N_attack+1:end)>=Thresh2)/N_attack;
        Detect_ratio_MLP2       = sum(r_deviation4(end-N_attack+1:end)>=Thresh2)/N_attack;
        
        % Precision
        ppv1 = out.logsout.getElement('PPV1').Values.Data; 
        ppv1_eta = out.logsout.getElement('PPV1_eta').Values.Data; 
        
        ppv2 = out.logsout.getElement('PPV2').Values.Data; 
        ppv2_eta = out.logsout.getElement('PPV2_eta').Values.Data;
        
        PPV1{iter1,iter2} = ppv1;
        PPV1_eta{iter1,iter2} = ppv1_eta;
        
        PPV2{iter1,iter2} = ppv2;
        PPV2_eta{iter1,iter2} = ppv2_eta;