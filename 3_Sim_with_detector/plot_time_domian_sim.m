%% plot file for time-fomain simulation example

load r.mat
load yc.mat
load time.mat

r_deviation2 = r{1,1};
r_deviation = r{1,2};
r_deviation4 = r{1,3};
r_deviation3 = r{1,4};

yc_deviation2 = yc{1,1};
yc_deviation = yc{1,2};
yc_deviation4 = yc{1,3};
yc_deviation3 = yc{1,4};
yc_deviation(1:83)=0;

LW = 2;
FS = 15;

figure (1)
subplot(2,1,1)
plot(time_vec,r_deviation2,'g',time_vec,r_deviation,'b',time_vec,r_deviation4,'r',time_vec,r_deviation3,'k','LineWidth',LW)
legend('MLP1','MLP1+pruning','MLP2','MLP2+pruning');
ylabel('\Delta r')
xlabel('Time instance')

ax = gca;
ax.LineWidth = LW-0.5;
ax.FontSize = FS;

subplot(2,1,2)
plot(time_vec,abs(yc_deviation2),'g','LineWidth',LW);
hold on, plot(time_vec,abs(yc_deviation),'b','LineWidth',LW);
hold on, plot(time_vec,abs(yc_deviation4),'r','LineWidth',LW);
hold on, plot(time_vec,abs(yc_deviation3),'k','LineWidth',LW);
legend('MLP1','MLP1+pruning','MLP2','MLP2+pruning');
ylabel('\Delta y')
xlabel('Time instance')

ax = gca;
ax.LineWidth = LW-0.5;
ax.FontSize = FS;


%%%%%%%%%%%%%
figure (2)
subplot(2,1,1)
plot(time_vec,r_deviation3,'k','LineWidth',LW)

ylabel('\Delta pr (%)')
xlabel('Time instance')

ax = gca;
ax.LineWidth = LW-0.5;
ax.FontSize = FS;

subplot(2,1,2)
plot(time_vec,yc_deviation3,'k','LineWidth',LW)

ylabel('\Delta py (%)')
xlabel('Time instance')

ax = gca;
ax.LineWidth = LW-0.5;
ax.FontSize = FS;
