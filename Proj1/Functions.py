#!/usr/bin/env python
# coding: utf-8

# In[13]:


import torch
import math
import datetime
import time
import numpy as np
from torch import optim
from torch import Tensor
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import torch.nn as nn
import dlc_practical_prologue      
import matplotlib.pyplot as plt
import warnings
from random import seed
from random import randint


# # Creating Convolutional network

# In[28]:


class Conv_Net(nn.Module):
    '''
    Creating the class for the convolutional net. The kernel size, stride and padding are chosen such as to have approximately 
    70 000 parameters in total.

    '''
    def __init__(self, nb_hidden):
        super(Conv_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=2),           #Convolutional layer         
            nn.ReLU(),                                                      #ReLU activation
            nn.MaxPool2d(kernel_size=3, stride=2))                          #Pooling layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.fc1 = nn.Linear(64, nb_hidden)                                #Fully Connected layers
        self.fc2 = nn.Linear(nb_hidden, 10)
        
        #Recreating all layers in order not to have weight sharing
        
        self.layer1_noWS = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2_noWS = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.fc1_noWS = nn.Linear(64, nb_hidden)  
        self.fc2_noWS = nn.Linear(nb_hidden, 10)
        
        self.layer1_Comp = nn.Linear(20, 100)                                #Layers for digit comparison
        self.relu = nn.ReLU()
        self.layerh_Comp = nn.Linear(100, 100)
        self.layer2_Comp = nn.Linear(100, 2)

    def forward(self, option_ws, train_input):
        img1 = train_input.narrow(1,0,1)                                      #Extracting the first image
        img2 = train_input.narrow(1,1,1)                                      #Extracting the second image
        
        x1 = self.layer1(img1)                                                #Processing the images in the neural network
        x1 = self.layer2(x1)
        x1 = x1.reshape(x1.size(0), -1)
        x1 = self.fc1(x1)
        x1 = self.fc2(x1)
        
        if option_ws == "no_weight_sharing":                            #No weight sharing: img2 uses different layers than img1
            x2 = self.layer1_noWS(img2)
            x2 = self.layer2_noWS(x2)
            x2 = x2.reshape(x2.size(0), -1)
            x2 = self.fc1_noWS(x2)
            x2 = self.fc2_noWS(x2)
        elif option_ws == "weight_sharing":                             #Weight sharing: img2 uses the same layers as img1
            x2 = self.layer1(img2)
            x2 = self.layer2(x2)
            x2 = x2.reshape(x2.size(0), -1)
            x2 = self.fc1(x2)
            x2 = self.fc2(x2)
        else :
            return "Please choose the type of convolutional network you would like to use: 'weight_sharing' or 'no_weight_sharing'."
        
        z = torch.cat((x1,x2),1)                                              #Concatenating x and y
        z = self.layer1_Comp(z)
        z = self.relu(z)
        z = self.layerh_Comp(z)
        z = self.relu(z)
        z = self.layer2_Comp(z)                                               #Result of the comparison
        
        
        return x1, x2, z


def count_parameters(model): 
    '''
    Function that counts the nuber of parameters in the convolutional net
    input: model - the neural network used
    output: nb_param - the number of parameters in the neural network
    ''' 
    nb_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return nb_param


# # Training the convolutional network

# In[23]:


def compute_error_rate(option_ws, option_err, model, input_img, input_target):
    '''
    Function that computes the error rate with an option to compute error rate of images or direct comparison.
    input:  option_err - choice for computing the error rate of the images or the prediction of the direct comparison
            option_ws - choice for using the neural network with or without weight sharing
            model - the neural network
            input_img - the two images
            input_target - the real comparison
    output: error_rate - the error rate as calculated according to the option chosen
    '''
    size = input_img.size(0)
    img1, img2, comp = model(option_ws, input_img)
    
    if option_err == "images":
        _, number1 = torch.max(img1, 1)
        _, number2 = torch.max(img2, 1)
        predicted = (number1 <= number2)
        error_rate = (torch.sum(input_target != predicted).item()/size)*100
        
    elif option_err == "outcome":
        _, isbigger = torch.max(comp, 1) 
        error_rate = (torch.sum(input_target != isbigger).item()/size)*100
    else :
        return "Please choose the type of error rate you would like to calculate: 'images' or 'outcome'."
    return error_rate


# In[30]:


def train_model(option_ws, option_loss, model, train_input, train_target, test_input, test_target,\
                batch_size, learning_rate, nb_epochs, train_classes, test_classes):
    '''
    Function that trains the neural network. Option to calculate the loss in an auxiliary manner or only based on the outcome.
    input:  option_ws - choice for using the neural network with or without weight sharing
            option_loss - choice for calculating the loss in an auxiliary manner or only based on the comparison
            model - the neural network
            train_input - the train input generated by the generate_pair_sets function
            train_target - the train target generated by the generate_pair_sets function
            test_input - the test input generated by the generate_pair_sets function
            test_target - the test target generated by the generate_pair_sets function
    output: loss_array - an array with all the calculated losses
            test_err1 - the test error for the comparison by images
            test_err2 - the test error for the direct comparison
            train_err1 - the train error for the comparison by images
            train_err2 - the train error for the direct comparison
    '''
    criterion = nn.CrossEntropyLoss()                         #Criterion for loss calculation
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    loss_array = []
    test_err1 = []                                            #Test error based on the images
    test_err2 = []                                            #Test error based on the outcome
    train_err1 = []                                           #Train error based on the images
    train_err2 = []                                           #Train error based on the outcome

    #Looping on the number of epochs
    for e in range(nb_epochs):
        start_time = time.time()                              #Calulating time for one round
        loss_sum = 0
        for b in range(0, train_input.size(0), batch_size):
            
            #Loss calculations using the chosen criterion
            img1, img2, comp = model(option_ws, train_input.narrow(0, b, batch_size))
            loss_img1 = criterion(img1, train_classes.narrow(0, b, batch_size).narrow(1,0,1).view(-1))
            loss_img2 = criterion(img2, train_classes.narrow(0, b, batch_size).narrow(1,1,1).view(-1))
            loss_comp = criterion(comp, train_target.narrow(0, b, batch_size))
            if option_loss == "auxiliary":
                loss_final = loss_img1 + loss_img2 + loss_comp                                    #Auxiliary loss  
            elif option_loss == "no_auxiliary":
                loss_final = loss_comp
            else:
                return "Please choose the type of loss you would like to use: 'auxiliary' or 'no_auxiliary'."
            model.zero_grad()
            loss_final.backward()
            optimizer.step()
            loss_sum += loss_final.item()
        loss_array.append(loss_sum)
        
        #Putting the error rates in the correspondent arrays
        
        test_err1.append(compute_error_rate(option_ws, "images", model, test_input, test_target))
        test_err2.append(compute_error_rate(option_ws, "outcome", model, test_input, test_target))
        train_err1.append(compute_error_rate(option_ws, "images", model, train_input, train_target))
        train_err2.append(compute_error_rate(option_ws, "outcome", model, train_input, train_target))
        
    end_time = time.time()
    time_elapsed = end_time-start_time
    print('\nElapsed time for training :{}'.format(time_elapsed))
    
    return loss_array, test_err1, test_err2, train_err1, train_err2


# # Printing and plotting values

# In[31]:


def print_values(nb_epochs, loss_array, test_err1, test_err2, train_err1, train_err2):
    '''
    Function that prints the error rate values for every round
    input:  nb_epochs - the number of epochs used for one round
            round - the current round
            loss_array - an array with all the calculated losses
            test_err1 - the test error for the comparison by images
            test_err2 - the test error for the direct comparison
            train_err1 - the train error for the comparison by images
            train_err2 - the train error for the direct comparison
    '''
    loss_array = np.array(loss_array).round(3)
    test_err1 = np.array(test_err1).round(3)
    test_err2 = np.array(test_err2).round(3)
    train_err1 = np.array(train_err1).round(3)
    train_err2 = np.array(train_err2).round(3)
    
    print('\nLoss array for {} epochs : \n'.format(nb_epochs))
    print(*loss_array, sep = " / ")
    print('\nTest error rate for images : \n')
    print(*test_err1, sep = " / " )
    print('\nTest error rate for outcome : \n')
    print(*test_err2, sep = " / " )
    print('\nTrain error rate for images \n')
    print(*train_err1, sep = " / " )
    print('\nTrain error rate for outcome : \n')
    print(*train_err2, sep = " / " )

def plot_error_rate(test_err1, test_err2, train_err1, train_err2):
    '''
    A function that plots the error rate over epochs
    input:  test_err1 - the test error for the comparison by images
            test_err2 - the test error for the direct comparison
            train_err1 - the train error for the comparison by images
            train_err2 - the train error for the direct comparison
    '''
    lines = plt.plot(test_err1, 'r', test_err2, 'r--', train_err1, 'b', train_err2, 'b--')
    plt.ylabel('Error rate')
    plt.xlabel('Epochs')
    plt.title("Test error in red, train error in blue")
    plt.legend(iter(lines), ('Test error images', 'Test error outcome', 'Train error images', 'Train error outcome'))
    plt.show()


# # Computing and printing mean and standard deviation

# In[32]:


def plot_results(test_err1_array, test_err2_array, train_err1_array, train_err2_array):
    '''
    A function that computes the mean standard deviation for the test and train errors
    input:  test_err1_array - the array of the test errors for the comparison by images
            test_err2_array - the array of the test errors for the direct comparison
            train_err1_array - the array of the train errors for the comparison by images
            train_err2_array - the array of the train errors for the direct comparison
    '''
    
    lines = plt.plot(test_err1_array, 'r', test_err2_array, 'r--', train_err1_array, 'b', train_err2_array, 'b--')
    plt.ylabel('Error rate')
    plt.xlabel('Rounds')
    plt.title("Test and train errors based on images and outcome")
    plt.legend(iter(lines), ('Test error images', 'Test error outcome', 'Train error images', 'Train error outcome'))
    plt.show()
    
    test_err1_mean = np.mean(test_err1_array)
    test_err2_mean = np.mean(test_err2_array)
    train_err1_mean = np.mean(train_err1_array)
    train_err2_mean = np.mean(train_err2_array)
    
    test_err1_std = np.std(test_err1_array)
    test_err2_std = np.std(test_err2_array)
    train_err1_std = np.std(train_err1_array)
    train_err2_std = np.std(train_err2_array)

    print('Test error based on images : \n mean : {:.3f} - standard deviation : {:.3f}'.format(test_err1_mean, test_err1_std))
    print('Test error based on outcome : \n mean : {:.3f} - standard deviation : {:.3f}'.format(test_err2_mean, test_err2_std))
    print('Train error based on images : \n mean : {:.3f} - standard deviation : {:.3f}'.format(train_err1_mean, train_err1_std))
    print('Train error based on outcome : \n mean : {:.3f} - standard deviation : {:.3f}'.format(train_err2_mean, train_err2_std))







