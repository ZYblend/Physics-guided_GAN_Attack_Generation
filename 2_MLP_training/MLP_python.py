'''
## Train MLP and get the trained weights
# sklearn.neural_network: MLPClassifier
##
# Yu Zheng, RASLab, FAMU-FSU College of Engineering, Tallahassee, 2021, Aug.
'''

## load packages
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
# from sklearn.model_selection import GridSearchCV   ## consider to add this later (https://coderzcolumn.com/tutorials/machine-learning/scikit-learn-sklearn-neural-network)
                                                      # Using cross validation to choose the best MLP structure and run training parallelly

## parameter
Input_size = 568   # T-horizon measurements and control inputs (8*(61+10))
Output_size = 61   # indix of attack support (size = number of measurement (61))

## Load training dataset
train_data = pd.read_csv('Dataset_for_MLP1.csv', header = None)
# train_data = pd.read_csv('Dataset_for_MLP2.csv', header = None)
train_data = np.array(train_data.values)
print(train_data.shape)                    # data struture: [N_data-by-(Input_size+Output_size)] train_data = [Input, Output]

# use last 5000 data as test dataset
start_test_data = train_data.shape[0]-5000
x_train = train_data[:start_test_data,:Input_size]
y_train = train_data[:start_test_data,Input_size:Input_size+Output_size]
x_test = train_data[start_test_data:train_data.shape[0],:Input_size]
y_test = train_data[start_test_data:train_data.shape[0],Input_size:Input_size+Output_size]
print(x_test.shape)
print(y_test.shape)

## MLP Training
clf = MLPClassifier(hidden_layer_sizes=(512,1024,512,),random_state=1, max_iter=30000).fit(x_train,y_train)

plt.plot(clf.loss_curve_)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

## save weights
w1 = clf.coefs_[0]
w2 = clf.coefs_[1]
w3 = clf.coefs_[2]
w4 = clf.coefs_[3]

np.savetxt('w1.csv',w1,delimiter=",")
np.savetxt('w2.csv',w2,delimiter=",")
np.savetxt('w3.csv',w3,delimiter=",")
np.savetxt('w4.csv',w4,delimiter=",")

b1 = clf.intercepts_[0]
b2 = clf.intercepts_[1]
b3 = clf.intercepts_[2]
b4 = clf.intercepts_[3]

np.savetxt('b1.csv',b1,delimiter=",")
np.savetxt('b2.csv',b2,delimiter=",")
np.savetxt('b3.csv',b3,delimiter=",")
np.savetxt('b4.csv',b4,delimiter=",")

## testing
test_output_prob = clf.predict_proba(x_test)
print(test_output_prob[4977,:])
# print(test_output_prob[1171,:])
test_output = clf.predict(x_test)
print(test_output[4977,:])
# print(test_output[1171,:])
print(y_test[4977,:])
print(clf.score(x_test, y_test))
print(precision_score(y_test,test_output,average='micro'))


print(clf.get_params(deep=True))