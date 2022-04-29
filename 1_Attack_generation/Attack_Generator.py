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
# import matplotlib.pyplot as plt
import os

class Attack_Generator:
    def __init__(self, M1,M2,I_attack):
        self.M1_tau = tf.cast(M1[:,I_attack],tf.float32)
        self.M2_tau = tf.cast(M2[:,I_attack],tf.float32)
        self.tau1 = 0.1
        self.tau2 = 0.5
        self.T = 8
        self.I_attack = I_attack

        self.num_iteration = 10000
        self.batch_size = 1000
        self.n_examples = 10000

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


    ## total training
    def Training(self):
        '''
        Main train function: 1. train generator, 
                             2. check if geneator trained successfully, 
                             3. generate tank of attacks
        '''
        # training
        Loss = np.zeros([self.num_iteration,1])
        for iter in range(1,self.num_iteration):
            self.loss = self.train_step()
            if iter % 999 ==0:
                tf.print('loss is', self.loss)
        Loss[iter-1] = self.loss.numpy()

        # check if generator is trained successfully
        if self.loss > 1 or np.isnan(self.loss):
            print('successful attacks cannot be found, please consider changing attack support')
            feasible_flag = 0
        else:
            feasible_flag = 1

        # generate attack tank
        test_input = tf.random.uniform([self.n_examples, self.n_inputs], minval=-1, maxval=1)
        xtest = self.model(test_input)
        self.attacks = xtest*self.am_scale
        y1_test = self.f1(self.attacks)    # effectiveness at next time step
        y2_test = self.f2(self.attacks)    # cumulative detection power in T horizon

        return Loss, self.attacks,feasible_flag, y1_test, y2_test

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

    ## attack scenarios
    n_attack_max = 30   # maximum number of attacks
    n_attack_seq = round(np.linspace(1,n_attack_max,num=n_attack_max,dtype='int'))  # seuqence for number of attacks (1,2,...,n_attack_max), corresponding index:(0,1,...,n_attack_max-1)
    n_meas = 61        # number of sensors
    meas_seq = np.linspace(0,n_meas-1,num = n_meas,dtype='int') # sequence for index of measurements (0,2,...,n_meas-1)


    ## train attack generator and collect attacks
    '''
    Data structure: 3-layer list
                      1. attack_tank[i]      : attack dataset corresponding to i+1 attacks (i=0,1,...,n_attack_max-1)
                      2. attack_tank[i][j]   : the No.j+1 dataset under i+1 attacks (j=0,1,..,tot_eva-1)
                      3. attack_tank[i][j][k]: k=1: attacks; k=2: attack support
    '''
    '''
    Since the python list data cannot be read by matlab, the above data struture would be translated to structure of folders
    3 -layers folders: attack_dataset/n_attack/i
                      1. attack_dataset
                      2. for different number of attacks
                      3. for different attack support
    '''
    tot_eva = 5     # number of different attack support under one of n_attack
    max_eva = 40    # maximum evaluation times
    attack_tank = []
    # parent_dir = "/home/digitalstorm/Documents/Yu_Zheng/Automated_attack_generator/attack_dataset"
    current_dir = os.getcwd()
    parent_dir = current_dir + "/attack_dataset"
    for idx in range(0,n_attack_max):   # from 0 to n_attack_max-1
        n_attack = n_attack_seq[idx]
        attack_tank_per_attack_num = []  # cache for saving attack_tank[i][j]

        # create layer2 folders
        dir1 = "attack"+str(n_attack)           
        path1 = os.path.join(parent_dir, dir1)
        os.mkdir(path1)

        for iter in range(1,tot_eva+1):    # from 1 to tot_eva
            feasible_flag = 0   # flag: if the attacks generated are feasible
            failflag = 0        # flag: if there is no feasible attack for the attack percentage
            num_try = 0
            while feasible_flag == 0:
                num_try = num_try+1
                print(str(num_try)+'th try')
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
                Loss, attacks, feasible_flag, y1_test, y2_test = attack_Generator.Training()  # train generator
                
                if num_try>max_eva:                                          # if there is no feasible attacks can be generated
                    print('Cannot find feasible attack support for', n_attack, 'attacks')
                    print('Please try to increase max_eva in order to search more or increase attack percentage')
                    failflag = 1                                             # return failflag=1, which means current attack support is feasible
                    break

            if failflag == 0 :
                attack_tank_per_attack_num.append([attacks,I_attack])

                # create layer3 folders and save data
                dir2 = str(iter)
                path2 = os.path.join(path1, dir2)
                os.mkdir(path2)
                path31 = os.path.join(path2, "attacks.csv")
                path32 = os.path.join(path2, "I_attack.csv")
                path33 = os.path.join(path2, "y1_effect.csv")
                path34 = os.path.join(path2, "y2_detect.csv")                
                np.savetxt(path31, attacks, delimiter=',')
                np.savetxt(path32, I_attack, delimiter=',')
                np.savetxt(path33, y1_test, delimiter=',')
                np.savetxt(path34, y2_test, delimiter=',')


        attack_tank.append(attack_tank_per_attack_num)
    
    
        
                

    








        

    
