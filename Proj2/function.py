
from torch import Tensor
import math


def convert_to_one_hot_labels(input, target):

    #taken from dlc_practicals, one hot encode the label
    tmp = input.new(target.size(0), target.max() + 1).fill_(-1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp


def generate_disc_set(nb,one_hot_labels=True):
    #generate the data in train and test set and labels them with one hot encoding
    input = Tensor(2, nb).uniform_(0, 1)
    center=Tensor(2,1).fill_(0.5)
    target=(input.sub(center).pow(2).sum(dim=0).sqrt()<1/math.sqrt(2*math.pi)).long()
    
    if(one_hot_labels):
        target = convert_to_one_hot_labels(input,target).t()

    return input, target

 
def train_model(model, train_set, train_label, loss, mini_batch_size, eta, n_epochs): 
    #train the model by minimizing loss
    n = train_set.size()[1]
    errors = []
    #iterate over number of epoch
    for e in range(0, n_epochs):
        loss_epoch=0
        
        #iterate over mini_batches
        for b in range( int(0) , int(train_set.size()[1]) , int(mini_batch_size) ):
            x_out=model.forward(train_set[:,b:b+mini_batch_size])
            loss_batch = loss.forward(x_out, train_label[:,b:b+mini_batch_size])
            loss_epoch=loss_epoch+loss_batch
            model.backward(loss.backward())
            
            #updating weights and bias
            for p in model.param():
                p.update_param(eta)
         
        print("Epoch: {} \t -> Loss: {} ".format(e, loss_epoch))  
        #adding the errors to the vector for one epoch 
        errors.append(loss_epoch)    
    return errors        

def err_classification(x_out, label):
    #counting the number of  classifiation errors 
    output_one_hot=x_out.max(dim=0)[1]
    label_one_hot=label.max(dim=0)[1]
    return (label_one_hot-output_one_hot).nonzero().shape[0]
    
   





