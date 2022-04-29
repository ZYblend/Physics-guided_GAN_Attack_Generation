import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



loss_temp = pd.read_csv('Single_attack_data/Loss.csv', header = None)
loss = np.array(loss_temp.values)
    
test_loss_temp = pd.read_csv('Single_attack_data/Test_Loss.csv', header = None)
test_loss = np.array(test_loss_temp.values)

plt.plot(loss[50:],label='train')
plt.plot(test_loss[50:],label='test')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.title('Loss Curve')
plt.show()

Y1_test_temp = pd.read_csv('Single_attack_data/Y1_test.csv', header = None)
Y1_test = np.array(Y1_test_temp.values)
print(Y1_test.shape)
    
Y2_test_temp = pd.read_csv('Single_attack_data/Y2_test.csv', header = None)
Y2_test = np.array(Y2_test_temp.values)

ITERATION = loss.shape[0]
tau1 = 0.1   # for effect
tau2 = 0.5   # for detect

x_val_size = int((ITERATION)) #for all points: hp.ITERATION * hp.BATCH_SIZE
x_val = np.linspace(1, x_val_size-51, num=x_val_size-51)
x_val = x_val.reshape(x_val.shape[0], 1)


fig, ax = plt.subplots(1,2)
for iter in range(20):
    ax[0].scatter(x_val, Y1_test[51:,iter], c = "k",marker='.')
    ax[1].scatter(x_val, Y2_test[51:,iter], c = "k",marker='.')


ax[0].plot(x_val, tau1*np.ones(x_val.shape),'r', label="alpha")
ax[1].plot(x_val, tau2*np.ones(x_val.shape),'r', label="epsilon")
ax[0].set_title('effectiveness')
ax[1].set_title('stealthiness')
ax[0].set_xlabel('Epoch')
ax[1].set_xlabel('Epoch')
ax[0].legend()
ax[1].legend()
plt.show()