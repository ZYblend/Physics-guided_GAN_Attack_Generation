'''
This is class for generative model
Description:
            Given system effectiveness matrix M1 and system detection matrix M2,
               Generate T-horizon attack e corresponding to attack support I_attack by solving
               ||M1(:,I_attack)e||_2 >= tau1  (effectiveness at next time step)
               ||M2(:,I_attack)e||_2 <= tau2  (cumulative detection power in T horizon)
             A generative neural network is used to minimize:
                Loss = relu(tau1-||M1(:,s)e||_2 ) + relu(||M2(:,s)e||_2 - tau2 ) 
Hyperparameters:
             - tau1, tau2: [scalar] thresholds for detection and effectiveness
             - M1: [n_m1-by-T*n] system effectiveness matrix 
             - M1: [n_m2-by-T*n] system detection matrix 
             - T: [scalar] time horizon
             - num_iteration: [scalar] maximum number of training iterations
             - batch_size: [scalar] 
             - am_scale: [scalar] amplifier factor for attack magtitude 
Inputs: 
         - I_attack: [T*n_attack-by-1] attack support for T-horizon
outputs: 
         - attacks: [n_examples-by-n_attack] a tank of generated attacks
         - feasible_flag: [scalar] flag of successful generation (1:success, 0:fail)
...................
Yu Zheng, RASLab, FAMU-FSU College of Engineering, Tallahassee, 2021, Aug.
...................
'''

## load packages
import pandas as pd
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold

class Attack_Generator:
    def __init__(self, M1,M2,I_attack):
        self.M1_tau = tf.cast(M1[:,I_attack],tf.float32)
        self.M2_tau = tf.cast(M2[:,I_attack],tf.float32)
        self.tau1 = 0.1 # 0.5  # for effect
        self.tau2 = 0.5 # 1    # for detect
        self.T = 8
        self.I_attack = I_attack

        self.num_iteration = 5000
        self.batch_size = 1000
        self.n_examples = 20

        self.am_scale = 10
    

    ## Define generator
    def define_generator(self):
        '''
        1. Based on Box-Muller Transform, 2 uniform signal can produce 2 normal distributed signal
            Thus, the minimal input size should satisfy: C^2_{n_inputs} = n_outputs 
             To simplify, we set input size equal to 8
        2. Output layer uses sigmoid function in order to give a interval to search feasible attacks
            instead of searching globally
        '''
        n_outputs = self.I_attack.shape[0]
        self.n_inputs = 8

        self.model = tf.keras.Sequential([
            layers.Dense(2*self.n_inputs, activation='relu', kernel_initializer='he_uniform', input_dim=self.n_inputs),
            layers.Dense(n_outputs, activation='sigmoid') 
            ])
        return self.model
    

    ## Loss function
    def loss_fcn(self,xhat):
        '''
        Loss function: relu(tau1-||M1(:,s)e||_2 ) + relu(||M2(:,s)e||_2 - tau2 ) 
        Inputs: 
               - xhat: [T*n_attack,1]: output of generator
        '''
        # target function for physical-based discriminator
        self.f1 = lambda x: tf.norm(tf.matmul(self.M1_tau,tf.transpose(x)),axis=0, keepdims=True) # M1(:,s)e effectiveness
        self.f2 = lambda x: tf.norm(tf.matmul(self.M2_tau,tf.transpose(x)),axis=0, keepdims=True) # M2(:,s)e cumulative detection

        # get attack by amplifying output of generator
        attack = xhat*self.am_scale
        
        # Prepare thresholds for whole batch
        Tau1 = self.tau1*np.ones(shape=[self.batch_size,])
        Tau2 = self.tau2*np.ones(shape=[self.batch_size,])
        thresh1 = tf.cast(tf.convert_to_tensor(Tau1), tf.float32)
        thresh2 = tf.cast(tf.convert_to_tensor(Tau2), tf.float32)
        
        # calculate Loss
        loss1 = tf.nn.relu(thresh1 - tf.transpose(self.f1(attack)))
        loss2 = tf.nn.relu(tf.transpose(self.f2(attack)) - thresh2)
        
        loss = tf.divide(tf.reduce_sum(loss1+loss2),self.batch_size)
        
        return loss
    
    
    ## Training for one step
    def train_step(self):
        '''
        train by back propogation
        '''
        # generate latent space inputs
        inputs = tf.random.uniform([self.batch_size, self.n_inputs], minval=-1, maxval=1)

        # define optimizer
        optim = tf.keras.optimizers.Adam(0.01, beta_1=0.5, beta_2=0.999)
        # define gradient function
        with tf.GradientTape() as t_tape:
            x_hat = self.model(inputs, training=True)
            loss= self.loss_fcn(x_hat)
            
        # calculate gradients
        t_gradients = t_tape.gradient(loss, self.model.trainable_variables)

        # update network weights
        optim.apply_gradients(zip(t_gradients, self.model.trainable_variables))
        return loss
    
    def test_step(self):
        test_input = tf.random.uniform([self.n_examples, self.n_inputs], minval=-1, maxval=1)
        xtest = self.model(test_input)
        test_loss = self.loss_fcn(xtest)
        self.attacks = xtest*self.am_scale
        y1_test = self.f1(self.attacks)    # effectiveness at next time step
        y2_test = self.f2(self.attacks)    # cumulative detection power in T horizon
        return test_loss, y1_test.numpy(), y2_test.numpy()


    ## total training
    def Training(self):
        '''
        Main train function: 1. train generator, 
                             2. check if geneator trained successfully, 
                             3. generate tank of attacks
        '''
        # training
        Loss = np.zeros([self.num_iteration,1])
        Test_Loss = np.zeros([self.num_iteration,1])
        Y1_test = np.zeros([1,self.n_examples])
        Y2_test = np.zeros([1,self.n_examples])

        for iter in range(1,self.num_iteration):
            self.loss = self.train_step()
            if iter % 99 ==0:
                tf.print('loss is', self.loss)
            test_loss, y1_test, y2_test = self.test_step()

            Loss[iter-1] = self.loss.numpy()
            Test_Loss[iter-1] = test_loss.numpy()

            Y1_test = np.concatenate((Y1_test,y1_test),axis=0)
            Y2_test = np.concatenate((Y2_test,y2_test),axis=0)

        # check if generator is trained successfully
        if self.loss > 1 or np.isnan(self.loss):
            print('successful attacks cannot be found, please consider changing attack support by re-running this program')
            feasible_flag = 0
        else:
            feasible_flag = 1

        # generate attack tank
        test_input = tf.random.uniform([self.n_examples*500, self.n_inputs], minval=-1, maxval=1)
        xtest = self.model(test_input)
        self.attacks = xtest*self.am_scale
        y1_test = self.f1(self.attacks)    # effectiveness at next time step
        y2_test = self.f2(self.attacks)    # cumulative detection power in T horizon

        return Loss, Test_Loss, self.attacks,feasible_flag, y1_test, y2_test, Y1_test, Y2_test

# ############################## MAIN ######################################
'''
Generate attacks for different attack support under different attack percentage


'''
if __name__ == '__main__':
    ## Load data 
    M1_temp = pd.read_csv('M1.csv', header = None)
    M1 = np.array(M1_temp.values)
    
    M2_temp = pd.read_csv('M2.csv', header = None)
    M2 = np.array(M2_temp.values)

    T = 8  # time horizon

    n_attack = 24
    n_meas = 61        # number of sensors
    meas_seq = np.linspace(0,n_meas-1,num = n_meas,dtype='int') # sequence for index of measurements (0,2,...,n_meas-1)

    # Create random attack support
    I_attack_local = np.random.choice(meas_seq,n_attack,replace=False)   
    I_attack =  np.zeros([n_attack*T,])
    for idx in range(1,T+1):
        I_attack[(idx-1)*n_attack:idx*n_attack] = I_attack_local+(idx-1)*n_meas
    I_attack = np.sort(I_attack)
    I_attack = I_attack.astype(int)

    # generator training
    attack_Generator = Attack_Generator( M1,M2,I_attack)    # create generative network
    generator = attack_Generator.define_generator()         
    Loss, Test_Loss, attacks, feasible_flag, y1_test, y2_test, Y1_test, Y2_test = attack_Generator.Training()  # train generator\

    print(attacks)
    print(y1_test)
    print(y2_test)

    np.savetxt("tau1.csv", np.array([attack_Generator.tau1]))
    np.savetxt("tau2.csv", np.array([attack_Generator.tau2]))
    np.savetxt("Single_attack_data/attacks.csv", attacks.numpy(), delimiter=',')
    np.savetxt("Single_attack_data/I_attack.csv", I_attack, delimiter=',')
    np.savetxt("Single_attack_data/y1_effect.csv", y1_test.numpy(), delimiter=',')
    np.savetxt("Single_attack_data/y2_detect.csv", y2_test.numpy(), delimiter=',')
    np.savetxt("Single_attack_data/Loss.csv",Loss,delimiter=',')
    np.savetxt("Single_attack_data/Test_Loss.csv",Test_Loss,delimiter=',')
    np.savetxt("Single_attack_data/Y1_test.csv",Y1_test,delimiter=',')
    np.savetxt("Single_attack_data/Y2_test.csv",Y2_test,delimiter=',')

    # Plotting
    plt.plot(Loss)
    plt.plot(Test_Loss)
    plt.show()
    
    
        
                

    








        

    
