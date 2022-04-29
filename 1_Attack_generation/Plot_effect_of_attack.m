r = simOut.logsout.getElement('r').Values.Data;
r_attacked = simOut.logsout.getElement('r_attacked').Values.Data;

x               = simOut.logsout.getElement('x').Values.Data;
x_attacked      = simOut.logsout.getElement('x_attacked').Values.Data;

yc = simOut.logsout.getElement('yc').Values.Data;
yc_attacked = simOut.logsout.getElement('yc_attacked').Values.Data;

LW = 1.5;  % linewidth
FS = 13;   % font size

figure (2)
subplot(1,2,1)
plot(r(84:end,:),'.','LineWidth',2*LW)   % attack is injected from 84 time step
hold on, plot(r_attacked(84:end,:),'.','LineWidth',2*LW)
xlabel('Time','Fontsize',FS)
ylabel('Residual','Fontsize',FS)
legend('Nominal','Attacked','Fontsize',FS)
ax = gca;
ax.FontSize = FS; 
ax.LineWidth = LW;

subplot(1,2,2)
plot(yc(84:end,:),'.','LineWidth',2*LW)
hold on, plot(yc_attacked(84:end,:),'.','LineWidth',2*LW)
xlabel('Time','Fontsize',FS)
ylabel('Critical Measurements','Fontsize',FS)
legend('Nominal','Attacked','Fontsize',FS)
ax = gca;
ax.FontSize = FS; 
ax.LineWidth = LW;


% subplot(2,2,3)
% plot(100*abs(r-r_attacked)./r,'k');  % Deviation from the nominal residual
% xlabel('Time','Fontsize',FS)
% ylabel('Deviation Ratio (%)','Fontsize',FS)
% ax = gca;
% ax.FontSize = FS; 
% 
% subplot(2,2,4)
% plot(100*abs(sum(x,2)-sum(x_attacked,2))./sum(x,2),'k');  % Deviation from the nominal critical measurement
% xlabel('Time','Fontsize',FS)
% ylabel('Deviation Ratio (%)','Fontsize',FS)
% xlim([0 800])
% ax = gca;
% ax.FontSize = FS; 
