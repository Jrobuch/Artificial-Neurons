
# coding: utf-8

# ## Drexel University
# ## CS-615: Deep Learning
# ## HW1
# ## John Obuch

##############################################################################################################################

#import requirements part 2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np

#import requirements part 3
import math, os, cv2, random, collections, importlib
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder #ok to use per Maryam (TA)
from collections import Counter
from matplotlib import pyplot as plt

#import requirements part 4
import math, os, cv2, random, collections, importlib
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder #ok to use per Maryam (TA)
from collections import Counter
from matplotlib import pyplot as plt

###############################################################################################################################

# ### Part1 Theory: _See LaTex PDF_
print("Part 1: See LaTex PDF document")

################################################################################################################################

# ### Part 2 Gradient Descent:

print("\nPART 2:")
print("Visualizing Gradient Decent: See produced figures...")

############# NOTES #######################
#J = (theta1 + theta2 -2)^2
#dJ/dtheta1 = 2theta1 + 2theta2 - 4
#dJ/dtheta2 = 2theta1 + 2theta2 - 4

#dJ/dtheta1 = dJ/dtheta2

#update: theta = theta - alpha*(dJ/dtheta)
###########################################

#initialize the values of theta
init_theta1 = 0
init_theta2 = 0

#define the learning rate
alpha = 0.1

#establish threshold conditions
prec = 1e-12
threshold = 10000

#define empty lists to store iteration updates of theta and J
theta1_L = []
theta2_L = []
J_L = []

#define functions for partial derivatives for theta
"""Note: Because the partial derivatives of our Cost Function with respect to theta1 and theta2 are equal,
and our initial conditions for theta1 and theta2 are equivelently set as zero,
the updates with respect to both parameters will achive equilent results. Thus, I will choose to combine
the results and simplify the cost function allowing to leverage only one partial function."""

def partial_theta(theta):
    
    """This function computes the resulting output after
    taking the partial derivative with respect to theta1
    of the function J = (theta1 + theta2 - 2)**2 however, 
    since the partials with respect to theta1 and theta2 will be the same, 
    I elect to simplify the cost function as follows: dJ/dtheta = 4*theta - 4"""
    
    return 4*theta - 4

#define the orginal cost function J
def cost_func_J(theta):
    
    """This function computes the resulting output 
    of the initial cost function J = (theta1 + theta2 - 2)**2
    however, since the updates to theta1 and theta2 will be the same
    based on their partials and our initial conditions of zeros, 
    I elect to simplify the cost function as follows: J = (2*theta - 2)**2"""
    
    return (2*theta - 2)**2

k = 0
for i in range(threshold):
    
    theta = init_theta1
    
    #perform the updates of theta
    init_theta1 = theta - alpha*partial_theta(theta)
    init_theta2 = init_theta1
    
    #compute the current J(theta1, theta2) result
    J = cost_func_J(theta)
    
    #append each iteration result to their associated lists (defined above)
    theta1_L.append(theta)
    theta2_L.append(theta)
    J_L.append(J)
    
    #compute the change in theta step sizes 
    #(Note: recall the change for both parameters step size will be the same based on our initial conditions)
    chng_in_theta1 = abs(init_theta1 - theta)
    
    #incriment the iteration count
    k += 1
    
    #apply a condition that terminates if the step size of the theta update to that of the previous theta is withing .000001
    if chng_in_theta1 <= prec:
        break
    
#cast the lists to numpy arrays (i.e. make them vectors) to feed ploting functions
t1 = np.array(theta1_L)
t2 = np.array(theta2_L)
j = np.array(J_L)

print('Theta converges to its true value after approximently {} iterations'.format(k))

#visualize the gradient decent process
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.plot(t1, t2, j, 'b')
# ax = plt.axes(projection='3d')
# ax.plot3D(t1, t2, j, 'b')
_ = plt.title("Visualizing Gradient Decent\n")
ax.set_xlabel(r'$\theta_1$')
ax.set_ylabel(r'$\theta_2$')
ax.set_zlabel(r'$J(\theta_1, \theta_2)$')
# plt.show(block=True)
plt.savefig('gradient_plot_3D')

###########################################################################
#CITATION: 
#This approach stemmed from refferencing wikipedias gradient decent page
#https://en.wikipedia.org/wiki/Gradient_descent
###########################################################################

###################################################################################################################################

# ### Part 3 Gradient Descent Logistic Regression:

print("\nPART 3:")
print("Training the data. Please wait...")

#establish the directory where the files are located
trainDirectory = "yalefaces/"

#store all files in the /yalefaces directory into a list (NOTE: The README.txt file is an element in the list!)
lFileList = []
for fFileObj in os.walk(trainDirectory): 
    lFileList = fFileObj[2]
    break  

#define empty lists to store flattened image row vectors into as well as the class value labels
matrix = []
class_ = []

#obtain X matrix data and Y target vector (i.e. class values)
for file in lFileList:
    
    #if file is not the README.txt file
    if file != 'Readme.txt':
        
        #read in the image, cast to numpy array and resize the image to be dimensions 40x40, and flatten the image
        im = cv2.resize(np.array(Image.open(trainDirectory+file)), (40,40)).flatten()
        
        #reshape the flattened images and append the images to the matrix list
        im = np.reshape(im, (1,im.shape[0]))[0]
        
        #append each row vector to the matrix list
        matrix.append(im)
        
        #split the file and grab first element subject<id>
        subject_id = file.split(".")[0]
        
        #grab the last two values in the string (i.e. the <id> number) and cast it to an integer
        class_val = int(subject_id[-2:])
        
        #append the values to the class_ list
        class_.append(class_val)

#cast the matrix list and class list to a numpy arrays (matrix/vector)
X = np.array(matrix) 
Y= np.array(np.reshape(class_, (len(class_), 1))) 

#concatinate the y vector to the x matrix such that the last column in matrix A is the target values
A = np.concatenate((X, Y), axis=1)

#initialize random number generator
np.random.seed(42)   ####Maryam says to average the accuracies without random seed works with seed of 42!!!!

#create list to store sub matricies associated to each class into
class_matrix_L = []

#get the sub-matricies for each class
for c in set(A[:, -1]):
    mask = A[:, -1] == c
    A_c = A[mask]
    class_matrix_L.append(A_c)

#define empty lists to store the rows of train/test sub-matries into to built the training and testing matricies
train_temp = []
test_temp = []

#shuffle the sub-matrices and split them into 2/3 train and 1/3 test split to ensure equal class prior probabilities
for mat in class_matrix_L:
    np.random.shuffle(mat)
    cutoff = math.ceil((2/3)*mat.shape[0])
    train_ = mat[0:cutoff, :]
    test_ = mat[cutoff:, :]
    
    #for each row in the train and test sub-matricies, append each row to the train_temp & test_temp lists
    for row in train_:
        train_temp.append(row)    
    for row in test_:
        test_temp.append(row)
    
#cast the train and test lists to numpy array matricies
A_train = np.array(train_temp) #numpy.ndarray
A_test = np.array(test_temp)

#randomize the rows in the train and test Matrix groups
np.random.shuffle(A_train)
np.random.shuffle(A_test)

#establish train and test groups
X_train = np.array(A_train[:, 0:-1]) #train
Y_train = np.array(A_train[: , -1])
X_test = np.array(A_test[: , 0:-1]) #test
Y_test = np.array(A_test[: , -1])

#one-hot-encode the Y_train and Y_test vectors to be matricies where each column is representative of a class. 
#per Maryam (TA) ok to use sklearn onehot encoder!
enc = OneHotEncoder(categories='auto')
Y_train_onehot = enc.fit_transform(Y_train.reshape(-1, 1)).toarray()
Y_test_onehot = enc.fit_transform(Y_test.reshape(-1, 1)).toarray()

#compute mean and std of training data to identify if there is a zero in the std vector
X_bar_train = np.mean(X_train, axis = 0)
X_std_train = np.std(X_train, axis = 0, ddof = 1)  

#remove the features that corresponded to std of zero
for indx, i in enumerate(range(len(X_std_train))):
    if X_std_train[i] == 0:
        X_train= np.delete(X_train, indx, axis=1)
        X_test=np.delete(X_test, indx, axis=1)

#create vector of ones (i.e. create bias feature for both train/test groups)
bias_feature_train = np.ones(X_train.shape[0])
bias_feature_test = np.ones(X_test.shape[0])

#recompute mean and std of training data
X_bar_train = np.mean(X_train, axis = 0)
X_std_train = np.std(X_train, axis = 0, ddof = 1)

#standardize the data
X_stdz_train = (X_train - X_bar_train)/X_std_train 
X_stdz_test = (X_test - X_bar_train)/X_std_train

#Define X matrix arrays (including bias feature)
X_stdz_train =np.column_stack([bias_feature_train, X_stdz_train[:, 0:X_stdz_train.shape[1]]])
X_stdz_test =np.column_stack([bias_feature_test, X_stdz_test[:, 0:X_stdz_test.shape[1]]]) 

#initialize theta matrix
theta = np.random.uniform(-1,1,(X_stdz_train.shape[1],Y_train_onehot.shape[1]))

# #define the learning rate Eta
eta = .01 #CHANGED ETA from .01 to 1 and applied eta/N and accuracy increased!

# #since performing Batch Gradient Decent we need to compute eta/N where N here is the size (number of records) of X_train
eta_over_N = eta/X_stdz_train.shape[0]

# #set the threshold values so we know when to stop
log_thresh = 2e-23
iter_thresh = 5000

# #set values of variables to compare to threshold values
chg_in_log = 1
prev_log = 1000

# #initialize the iteration count
iter_ = 0

#define lambda to be the regularization scalar factor
lambda_ = 5

#define empty lists to store iteration and cost history into
J_train = []
J_test = []

#perform Batch Gradient Decent until convergence criteria is met
while (chg_in_log > log_thresh) and (iter_ < iter_thresh):
    
    #define the L2 norm regularization term
    L2 = np.linalg.norm(theta) 
    
    ###### TEST DATA #######
    
    #compute the activation function output
    Y_hat_test = 1/(1 + np.exp(np.dot(-X_stdz_test, theta)))  

    #compute the error
    Y_err = (Y_test_onehot - Y_hat_test)

    #compute the current loss of the cost function for the testing data
    crnt_log_test = (1/X_stdz_test.shape[0])*(np.sum((Y_test_onehot) * np.log(Y_hat_test + 1e-5) + 
                                                      (1-Y_test_onehot) * np.log(1 - Y_hat_test + 1e-5)) - lambda_*L2)  
    #append the iterations and costs to the list
    J_test.append((iter_, crnt_log_test))
    
    ###### TRAIN DATA #######
    
    #compute y_hat via the activation function
    Y_hat_train = 1/(1 + np.exp(np.dot(-X_stdz_train,theta)))  
    
    #compute the error (i.e. the residuals)
    Y_err = (Y_train_onehot - Y_hat_train)

    #update the current loss of the cost function for the trianing data
    crnt_log_train = (1/X_stdz_train.shape[0])*(np.sum((Y_train_onehot) * np.log(Y_hat_train + 1e-5) + 
                                                       (1-Y_train_onehot) * np.log(1-Y_hat_train + 1e-5)) - lambda_*L2) 
    
    #append the iterations and costs to the list
    J_train.append((iter_,crnt_log_train))

    #compute the gradient
    grad = np.dot(X_stdz_train.T, Y_err) - lambda_*theta   #ASK TA!!!!

    #perform the update of theta
    theta = theta + eta_over_N*grad  

    #make the learning rate dynamic (smooths out convergence graph)
#     if (crnt_log_train - prev_log) < 0:
#         eta = .5*eta
        
    #update the change in loss and the previous loss
    chg_in_log = abs((prev_log - crnt_log_train))
    prev_log = crnt_log_train

    #increment the iteration count
    iter_ += 1
    
print("TOTAL ITERATIONS")
print(iter_)

#compute the probabilities of the classification for the testing training sets
Y_hat_test = 1/(1 + np.exp(np.dot(-X_stdz_test, theta)))
Y_hat_train =1/(1 + np.exp(np.dot(-X_stdz_train,theta)))

#creating an empty Y_train_prediction matrix where each elelement is zero
Y_hat_test_pred = np.zeros([Y_hat_test.shape[0], Y_hat_test.shape[1]])  #Test
for i, row in enumerate(Y_hat_test):
    Y_hat_test_pred[i][np.argmax(row)] = 1
            
#creating an empty Y_train_prediction matrix where each elelement is zero
Y_hat_train_pred = np.zeros([Y_hat_train.shape[0], Y_hat_train.shape[1]])  #Train
for i, row in enumerate(Y_hat_train):
    Y_hat_train_pred[i][np.argmax(row)] = 1

#keep track of iteration and cost for plots
cost_train = []
iteration_train = []
for tup in J_train:
    cost_train.append(tup[1])
    iteration_train.append(tup[0])
    
cost_test = []
iteration_test = []
for tup in J_test:
    cost_test.append(tup[1])
    iteration_test.append(tup[0])

#confusion matrix train
print("\nTRAIN: CONFUSION MATRIX\n")
conf_train = Y_train_onehot.T @ Y_hat_train_pred
print(conf_train)
    
#confustion matrix test
print("\nTEST: CONFUSION MATRIX\n")
conf_test = Y_test_onehot.T @ Y_hat_test_pred    ##As long as you keep track of what axis represents predicted vs actual. Note that X @ Y is the same as np.dot(X,Y) when performing a dot-product
print(conf_test)

#compute the accuracy of the systems
print("\nTRAIN ACCURACY:")
acc_train = np.trace(conf_train)/np.sum(conf_train)
print(acc_train)

print("\nTEST ACCURACY:")
acc_test = np.trace(conf_test)/np.sum(conf_test)
print(acc_test)

#plot the training cost outputs over the iterations
x_train = np.array(iteration_train)
y_train = np.array(cost_train)

fig, axes = plt.subplots()
axes.plot(x_train, y_train, 'b')
axes.set_title("Training Set Cost by Iteration Number")
axes.set_xlabel('Iteration')
axes.set_ylabel('Cost')
_ = plt.savefig("convergence_train")
# plt.show()

#plot the testing cost outputs over the iterations
x_test = np.array(iteration_test)
y_test = np.array(cost_test)

fig, axes = plt.subplots()
axes.plot(x_train, y_train, 'r')
axes.set_title("Testing Set Cost by Iteration Number")
axes.set_xlabel('Iteration')
axes.set_ylabel('Cost')
_ = plt.savefig("convergence_test")
# plt.show()
    
################################################################################################################################

# ### Part 4 Gradient Descent w/ Softmax and Cross-Entropy:

# coding: utf-8
print("\nPART 4:")
print("Training the data. Please wait...")

#establish the directory where the files are located
trainDirectory = "yalefaces/"

#store all files in the /yalefaces directory into a list (NOTE: The README.txt file is an element in the list!)
lFileList = []
for fFileObj in os.walk(trainDirectory): 
    lFileList = fFileObj[2]
    break  

#define empty lists to store flattened image row vectors into as well as the class value labels
matrix = []
class_ = []

#obtain X matrix data and Y target vector (i.e. class values)
for file in lFileList:
    
    #if file is not the README.txt file
    if file != 'Readme.txt':
        
        #read in the image, cast to numpy array and resize the image to be dimensions 40x40, and flatten the image
        im = cv2.resize(np.array(Image.open(trainDirectory+file)), (40,40)).flatten()
        
        #reshape the flattened images and append the images to the matrix list
        im = np.reshape(im, (1,im.shape[0]))[0]
        
        #append each row vector to the matrix list
        matrix.append(im)
        
        #split the file and grab first element subject<id>
        subject_id = file.split(".")[0]
        
        #grab the last two values in the string (i.e. the <id> number) and cast it to an integer
        class_val = int(subject_id[-2:])
        
        #append the values to the class_ list
        class_.append(class_val)

#cast the matrix list and class list to a numpy arrays (matrix/vector)
X = np.array(matrix) 
Y= np.array(np.reshape(class_, (len(class_), 1))) 

#concatinate the y vector to the x matrix such that the last column in matrix A is the target values
A = np.concatenate((X, Y), axis=1)

#initialize random number generator
np.random.seed(42)

#create list to store sub matricies associated to each class into
class_matrix_L = []

#get the sub-matricies for each class
for c in set(A[:, -1]):
    mask = A[:, -1] == c
    A_c = A[mask]
    class_matrix_L.append(A_c)

#get the class priors incase we need to use them
priors = []
for mat in class_matrix_L:
    priors.append(mat.shape[0]/A.shape[0])

#define empty lists to store the rows of train/test sub-matries into to built the training and testing matricies
train_temp = []
test_temp = []

#shuffle the sub-matrices and split them into 2/3 train and 1/3 test split to ensure equal class prior probabilities
for mat in class_matrix_L:
    np.random.shuffle(mat)
    cutoff = math.ceil((2/3)*mat.shape[0])
    train_ = mat[0:cutoff, :]
    test_ = mat[cutoff:, :]
    
    #for each row in the train and test sub-matricies, append each row to the train_temp & test_temp lists
    for row in train_:
        train_temp.append(row)    
    for row in test_:
        test_temp.append(row)
    
#cast the train and test lists to numpy array matricies
A_train = np.array(train_temp) #numpy.ndarray
A_test = np.array(test_temp)

#randomize the rows in the train and test Matrix groups
np.random.shuffle(A_train)
np.random.shuffle(A_test)

#establish train and test groups
X_train = np.array(A_train[:, 0:-1]) #train
Y_train = np.array(A_train[: , -1])
X_test = np.array(A_test[: , 0:-1]) #test
Y_test = np.array(A_test[: , -1])

#one-hot-encode the Y_train and Y_test vectors to be matricies where each column is representative of a class. 
#per Maryan (TA) ok to use sklearn onehot encoder!!! THIS RUNS ON TUX!!
enc = OneHotEncoder(categories = 'auto')
Y_train_onehot = enc.fit_transform(Y_train.reshape(-1, 1)).toarray()
Y_test_onehot = enc.fit_transform(Y_test.reshape(-1, 1)).toarray()

#compute mean and std of training data to identify if there is a zero in the std vector
X_bar_train = np.mean(X_train, axis = 0)
X_std_train = np.std(X_train, axis = 0, ddof = 1)  

#remove the features that corresponded to std of zero
for indx, i in enumerate(range(len(X_std_train))):
    if X_std_train[i] == 0:
        X_train= np.delete(X_train, indx, axis=1)
        X_test=np.delete(X_test, indx, axis=1)

#create vector of ones (i.e. create bias feature for both train/test groups)
bias_feature_train = np.ones(X_train.shape[0])
bias_feature_test = np.ones(X_test.shape[0])

#recompute mean and std of training data
X_bar_train = np.mean(X_train, axis = 0)
X_std_train = np.std(X_train, axis = 0, ddof = 1)

#standardize the data
X_stdz_train = (X_train - X_bar_train)/X_std_train 
X_stdz_test = (X_test - X_bar_train)/X_std_train

#Define X matrix arrays (including bias feature)
X_stdz_train =np.column_stack([bias_feature_train, X_stdz_train[:, 0:X_stdz_train.shape[1]]])
X_stdz_test =np.column_stack([bias_feature_test, X_stdz_test[:, 0:X_stdz_test.shape[1]]]) 

#initialize theta matrix
theta = np.random.uniform(-1,1,(X_stdz_train.shape[1],Y_train_onehot.shape[1]))

# #define the learning rate Eta
eta = .1

# #since performing Batch Gradient Decent we need to compute eta/N where N here is the size (number of records) of X_train
eta_over_N = eta/X_stdz_train.shape[0]  

# #set the threshold values so we know when to stop
log_thresh = 2e-23
iter_thresh = 50000

# #set values of variables to compare to threshold values
chg_in_cross_entropy = 1
prev_cross_entropy = 1000

# #initialize the iteration count
iter_ = 0

#define lambda to be the regularization scalar factor
lambda_ = 5

#define empty lists to store iteration and cost history into
J_train = []
J_test = []

#perform Batch Gradient Decent until convergence criteria is met
while (chg_in_cross_entropy > log_thresh) and (iter_ < iter_thresh):
    
    #define the L2 regularization term
    L2 = np.linalg.norm(theta) 

    #### TEST DATA #####
    
    g_x_test = np.dot(X_stdz_test, theta)
    
    #compute y_hat via the activation function
    Y_hat_test = np.exp(g_x_test)/np.reshape(np.sum(np.exp(g_x_test), axis = 1), (-1,1)) 
    
    #compute the error (i.e. the residuals)
    Y_err = (Y_hat_test - Y_test_onehot)
    
    #update the current cross entropy loss of the cost function for the trianing data
    crnt_cross_entropy_test = (1/X_stdz_test.shape[0])*(-np.sum(Y_test_onehot*np.log(Y_hat_test))) + lambda_*L2
    
    #append the iterations and costs to the list
    J_train.append((iter_, crnt_cross_entropy_test))
    
    ###### TRAIN DATA #######
    
    g_x_train = np.dot(X_stdz_train, theta)
    
    #TAKE CARE OF DIV BY ZERO (if applicable) and set them equal to the priors(j) (Note: N/A, all row observartion distributions sum to 1)
    
    #compute y_hat via the activation function 
    Y_hat_train = np.exp(g_x_train)/np.reshape(np.sum(np.exp(g_x_train), axis = 1), (-1,1))  
    
    #compute the error (i.e. the residuals)
    Y_err = (Y_hat_train - Y_train_onehot)
    
    #update the current cross entropy loss of the cost function for the trianing data
    crnt_cross_entropy_train= (1/X_stdz_train.shape[0])*(-np.sum(Y_train_onehot*np.log(Y_hat_train))) + lambda_*L2
    
    #append the iterations and costs to the list
    J_train.append((iter_, crnt_cross_entropy_train))
    
    #compute the batch gradient
    grad = np.dot(X_stdz_train.T, Y_err) + lambda_*theta   

    #perform the update of theta
    theta = theta - eta_over_N*grad  
    
      #make the learning rate dynamic
#     if (crnt_cross_entropy_train - prev_cross_entropy) < 0:
#         eta = .5*eta
        
    #update the change in loss and the previous loss
    chg_in_cross_entropy = abs((prev_cross_entropy - crnt_cross_entropy_train))
    prev_cross_entropy = crnt_cross_entropy_train

    #increment the iteration count
    iter_ += 1
    
print("TOTAL ITERATIONS")
print(iter_)

#compute the probabilities of the classification for the testing training sets
g_x_test = np.dot(X_stdz_test, theta)
g_x_train = np.dot(X_stdz_train, theta)

Y_hat_test = np.exp(g_x_test)/np.sum(np.exp(g_x_test))
Y_hat_train = np.exp(g_x_train)/np.sum(np.exp(g_x_train))

#creating an empty Y_train_prediction matrix where each elelement is zero
Y_hat_test_pred = np.zeros([Y_hat_test.shape[0], Y_hat_test.shape[1]])  #Test
for i, row in enumerate(Y_hat_test):
    Y_hat_test_pred[i][np.argmax(row)] = 1
            
#creating an empty Y_train_prediction matrix where each elelement is zero
Y_hat_train_pred = np.zeros([Y_hat_train.shape[0], Y_hat_train.shape[1]])  #Train
for i, row in enumerate(Y_hat_train):
    Y_hat_train_pred[i][np.argmax(row)] = 1

#keep track of iteration and cost for plots
cost_train = []
iteration_train = []
for tup in J_train:
    cost_train.append(tup[1])
    iteration_train.append(tup[0])
    
cost_test = []
iteration_test = []
for tup in J_test:
    cost_test.append(tup[1])
    iteration_test.append(tup[0])

#confusion matrix train
print("\nTRAIN: CONFUSION MATRIX")
conf_train = Y_train_onehot.T @ Y_hat_train_pred
print(conf_train)    

#confustion matrix test
print("\nTEST: CONFUSION MATRIX")
conf_test = Y_test_onehot.T @ Y_hat_test_pred   #X-axis --> True Class, Y-Axis --> Predicted Class
print(conf_test)

#compute the accuracy of the systems
print("\nTRAIN ACCURACY:")
acc_train = np.trace(conf_train)/np.sum(conf_train)
print(acc_train)

print("\nTEST ACCURACY:")
acc_test = np.trace(conf_test)/np.sum(conf_test)
print(acc_test)

#plot the training cost outputs over the iterations
x_train = np.array(iteration_train)
y_train = np.array(cost_train)

fig, axes = plt.subplots()
axes.plot(x_train, y_train, 'b')
axes.set_title("Training Set Cost by Iteration Number (Softmax)")
axes.set_xlabel('Iteration')
axes.set_ylabel('Cost')
_ = plt.savefig("convergence_train_softmax")
# plt.show()

#plot the testing cost outputs over the iterations
x_test = np.array(iteration_test)
y_test = np.array(cost_test)

fig, axes = plt.subplots()
axes.plot(x_train, y_train, 'r')
axes.set_title("Testing Set Cost by Iteration Number (Softmax)")
axes.set_xlabel('Iteration')
axes.set_ylabel('Cost')
_ = plt.savefig("convergence_test_softmax")
# plt.show()
