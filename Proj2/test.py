#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
import math 
import matplotlib.pyplot as plt


import module as mod
from function import convert_to_one_hot_labels, generate_disc_set, train_model, err_classification


# In[2]:


torch.set_grad_enabled( False )

#Generates a training and a test set of 1, 000 points sampled uniformly in [0, 1]2
#each with a label 0 if outside the disk centered at (0.5, 0.5) of radius 1/√2π, and 1 inside

#Number of sampled point
num_points= 1000

#Generating train and test sets with labels
train_set, train_label=generate_disc_set(num_points)
test_set, test_label=generate_disc_set(num_points)


# In[3]:


#Model chosen with layers, input and output dimension (Linear) and activation function (ReLU, Tanh)
model= mod.Sequential(mod.Linear(2,25), mod.ReLU(), mod.Linear(25,25), mod.ReLU(), mod.Linear(25,25), mod.ReLU(), mod.Linear(25,2), mod.Tanh())


# In[4]:


#Creation of object from MSELoss class to compute loss 
loss=mod.MSELoss()

#Choice of parameters to train the model
mini_batch_size=10
eta =0.01
n_epochs = 1000



# In[5]:


#Training the model
p=train_model(model, train_set, train_label, loss,mini_batch_size, eta , n_epochs)


# In[6]:


#Plotting training loss 
fig = plt.figure()
plt.plot(p) 
plt.title("Loss in training")
plt.xlabel("Number of epoch")
plt.ylabel("Loss accumalated")

#Computing of test errors

x_out = model.forward(test_set)
test_loss=loss.forward(x_out,test_label)
print('Test error = {}'.format(test_loss))

#Computing number of missclassified point
err_class=err_classification(x_out, test_label)
print('Missclassified points = {}/{}'.format(err_class, num_points))


# In[ ]:




