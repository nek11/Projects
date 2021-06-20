#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import math
import datetime
import time
import numpy as np
import dlc_practical_prologue      
import matplotlib.pyplot as plt
import warnings
from random import seed
from random import randint
from Functions import Conv_Net, count_parameters, compute_error_rate, train_model, print_values, plot_error_rate, plot_results


# In[2]:


nb_hidden = 150                        #Number of hidden layers
nb_epochs = 25                         #Number of epochs
learning_rate = 1e-3                   #Setting the learning rate for the optimizer
batch_size = 10                        #Size for batch
size = 1000                            #Numbers of digit pairs 
nb_rounds = 10                         #Number of rounds to run (1 round = 25 epochs)


# In[3]:


#Generating size pairs of digit images
train_input, train_target, train_classes,test_input, test_target, test_classes = dlc_practical_prologue.generate_pair_sets(size)

#Displaying some pseudo-random data
fig = plt.figure
k = randint(0,size)
for i in range(10):
    plt.subplot(2,10,i+1)
    plt.imshow(test_input[i+k][0], interpolation='gaussian')
    plt.title("{}".format(test_classes[i+k][0])) 
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,10,i+11)
    plt.imshow(test_input[i+k][1], interpolation='gaussian')
    plt.title("{}".format(test_classes[i+k][1]))
    plt.xticks([])
    plt.yticks([])


# In[ ]:


#ARCHITECTURE 1
#SIMPLE CONVOLUTIONAL NEURAL NETWORK WITHOUT WEIGHT SHARING AND AUXILIARY LOSS
print('\n---------------------------------')
print('\nArchitecture 1 : Convolutional neural network without weight sharing and auxiliary loss')
print('\n---------------------------------')

#Create a convolutional neural network
convnet = Conv_Net(nb_hidden)
nb_param = count_parameters(convnet)
print('\nNumber of parameters : {}'.format(nb_param))

test_err1_array = []
test_err2_array = []
train_err1_array = []
train_err2_array = []

for round in range(nb_rounds):
    print('\nRound {} :'.format(round))
    
    #Generating size pairs of digit images
    train_input, train_target, train_classes,    test_input, test_target, test_classes = dlc_practical_prologue.generate_pair_sets(size)
    
    #Train the model and compute error rate for test_input and train_input 
    #based on images and outcome
    loss_array, test_err1, test_err2, train_err1, train_err2 =    train_model("no_weight_sharing", "no_auxiliary", convnet, train_input, train_target, test_input, test_target,                batch_size, learning_rate, nb_epochs, train_classes, test_classes)
    
    #Uncomment the following line if you want to print loss and error arrays
    #print_values(nb_epochs, loss_array, test_err1, test_err2, train_err1, train_err2)
    
    #Add errors of last epoch in the arrays
    test_err1_array.append(test_err1.pop())
    test_err2_array.append(test_err2.pop())
    train_err1_array.append(train_err1.pop())
    train_err2_array.append(train_err2.pop())
    
    #Plot the results
    #plot_error_rate(test_err1, test_err2, train_err1, train_err2)

#Plot results and print mean and standard deviation
print('\nArchitecture 1 - Results for no weight sharing and no auxiliary loss :')
plot_results(test_err1_array, test_err2_array, train_err1_array, train_err2_array)


# In[ ]:


#ARCHITECTURE 2
#CONVOLUTIONAL NEURAL NETWORK WITH WEIGHT SHARING BUT WITHOUT AUXILIARY LOSS
print('\n---------------------------------')
print('\nArchitecture 2 : Convolutional neural network with weight sharing but without auxiliary loss')
print('\n---------------------------------')

#Create a convolutional neural network
convnet = Conv_Net(nb_hidden)
nb_param = count_parameters(convnet)
print('\nNumber of parameters : {}'.format(nb_param))

test_err1_array = []
test_err2_array = []
train_err1_array = []
train_err2_array = []

for round in range(nb_rounds):
    print('\nRound {} :'.format(round))
    
    #Generating size pairs of digit images
    train_input, train_target, train_classes,    test_input, test_target, test_classes = dlc_practical_prologue.generate_pair_sets(size)
    
    #Train the model and compute error rate for test_input and train_input 
    #based on images and outcome
    loss_array, test_err1, test_err2, train_err1, train_err2 =    train_model("weight_sharing", "no_auxiliary", convnet, train_input, train_target, test_input, test_target,                batch_size, learning_rate, nb_epochs, train_classes, test_classes)
    
    #Uncomment the following line if you want to print loss and error arrays
    #print_values(nb_epochs, loss_array, test_err1, test_err2, train_err1, train_err2)
    
    #Add errors of last epoch in the arrays
    test_err1_array.append(test_err1.pop())
    test_err2_array.append(test_err2.pop())
    train_err1_array.append(train_err1.pop())
    train_err2_array.append(train_err2.pop())
    
    #Plot the results
    plot_error_rate(test_err1, test_err2, train_err1, train_err2)

#Plot results and print mean and standard deviation
print('\nArchitecture 2 - Results for weight sharing but no auxiliary loss :')
plot_results(test_err1_array, test_err2_array, train_err1_array, train_err2_array)


# In[ ]:


#ARCHITECTURE 3
#CONVOLUTIONAL NEURAL NETWORK WITHOUT WEIGHT SHARING BUT WITH AUXILIARY LOSS
print('\n---------------------------------')
print('\nArchitecture 3 : Convolutional neural network without weight sharing but with auxiliary loss')
print('\n---------------------------------')

#Create a convolutional neural network
convnet = Conv_Net(nb_hidden)
nb_param = count_parameters(convnet)
print('\nNumber of parameters : {}'.format(nb_param))

test_err1_array = []
test_err2_array = []
train_err1_array = []
train_err2_array = []

for round in range(nb_rounds):
    print('\nRound {} :'.format(round))
    
    #Generating size pairs of digit images
    train_input, train_target, train_classes,    test_input, test_target, test_classes = dlc_practical_prologue.generate_pair_sets(size)
    
    #Train the model and compute error rate for test_input and train_input 
    #based on images and outcome
    loss_array, test_err1, test_err2, train_err1, train_err2 =    train_model("no_weight_sharing", "auxiliary", convnet, train_input, train_target, test_input, test_target,                batch_size, learning_rate, nb_epochs, train_classes, test_classes)
    
    #Uncomment the following line if you want to print loss and error arrays
    #print_values(nb_epochs, loss_array, test_err1, test_err2, train_err1, train_err2)
    
    #Add errors of last epoch in the arrays
    test_err1_array.append(test_err1.pop())
    test_err2_array.append(test_err2.pop())
    train_err1_array.append(train_err1.pop())
    train_err2_array.append(train_err2.pop())
    
    #Plot the results
    plot_error_rate(test_err1, test_err2, train_err1, train_err2)

#Plot results and print mean and standard deviation
print('\nArchitecture 3 - Results for no weight sharing but auxiliary loss :')
plot_results(test_err1_array, test_err2_array, train_err1_array, train_err2_array)


# In[ ]:


#ARCHITECTURE 4
#CONVOLUTIONAL NEURAL NETWORK WITH WEIGHT SHARING AND AUXILIARY LOSS
print('\n---------------------------------')
print('\nArchitecture 4 : Convolutional neural network with weight sharing and auxiliary loss')
print('\n---------------------------------')

#Create a convolutional neural network
convnet = Conv_Net(nb_hidden)
nb_param = count_parameters(convnet)
print('\nNumber of parameters : {}'.format(nb_param))

test_err1_array = []
test_err2_array = []
train_err1_array = []
train_err2_array = []

for round in range(nb_rounds):
    print('\nRound {} :'.format(round))
    
    #Generating size pairs of digit images
    train_input, train_target, train_classes,    test_input, test_target, test_classes = dlc_practical_prologue.generate_pair_sets(size)
    
    #Train the model and compute error rate for test_input and train_input 
    #based on images and outcome
    loss_array, test_err1, test_err2, train_err1, train_err2 =    train_model("weight_sharing", "auxiliary", convnet, train_input, train_target, test_input, test_target,                batch_size, learning_rate, nb_epochs, train_classes, test_classes)
    
    #Uncomment the following line if you want to print loss and error arrays
    #print_values(nb_epochs, loss_array, test_err1, test_err2, train_err1, train_err2)
    
    #Add errors of last epoch in the arrays
    test_err1_array.append(test_err1.pop())
    test_err2_array.append(test_err2.pop())
    train_err1_array.append(train_err1.pop())
    train_err2_array.append(train_err2.pop())
    
    #Plot the results
    plot_error_rate(test_err1, test_err2, train_err1, train_err2)

#Plot results and print mean and standard deviation
print('\nArchitecture 4 - Results for weight sharing and auxiliary loss :')
plot_results(test_err1_array, test_err2_array, train_err1_array, train_err2_array)


# In[ ]:





# In[ ]:




