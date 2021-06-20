
import torch
import math 


class Module ( object ) : 
   # Class contraining basic element for the model to learn: forward, backward and param
    
    def forward ( self , * input ) :
        raise NotImplementedError
        
    def backward ( self , * gradwrtoutput ) :
        raise NotImplementedError
        
    def param ( self ) :
        return []



# In[4]:


class Linear (Module):
    #linear Class layer inheriting from Module
    def __init__(self, in_dim, out_dim):
        
        #initialization of parameter at random to begin
        self.in_dim=in_dim
        self.out_dim=out_dim 
        
        #special multiplication by sqrt(2/in_dim)for xavier init to avoid vashing gradient 
        self.w=torch.empty(out_dim, in_dim).normal_(0, math.sqrt(2/in_dim))
        self.b=torch.empty(out_dim, 1).normal_(0, math.sqrt(2/in_dim))
 
        self.gradw= torch.empty(out_dim,in_dim).zero_()
        self.gradb = torch.empty(out_dim,1).zero_()
      
    def forward(self,x):
        
        #forward should get for input and returns, a tensor or a tuple of tensors
        #save the input
        self.input=x
        y=self.w.mm(self.input)+self.b 
    
        return y
    
    def backward(self , gradwrtoutput):
        
        #backward should get as input a tensor or a tuple of tensors containing the gradient of the loss with respect to the module’s output, 
        #update the gradient of weight and biais
        self.gradw=gradwrtoutput.mm(self.input.t())
        self.gradb=gradwrtoutput.sum(1).view(self.gradb.size()[0], self.gradb.size()[1])

        return self.w.t().mm(gradwrtoutput) 
    
    
    def update_param(self, eta): 
        #update the weight with learning rate eta and gradient of each parameters
        self.w=self.w-(eta*self.gradw)
        self.b=self.b-(eta*self.gradb)
     
    def param(self):
        #param should return a list of pairs, each composed of a parameter tensor, and a gradient tensor of same size 
        return [(self.w, self.gradw), (self.b, self.gradb)]

class Tanh(Module):
    
    #Tanh activation function 
    def __init__(self): 
        self.input=None 
    
    def forward(self,x):
        #save the input for backward 
        self.input=x
        y=self.input.tanh()
        return y 
    
    def backward (self,grad): 
        return  grad.mul(4 * (self.input.exp() + self.input.mul(-1).exp()).pow(-2)) 
    
    def param(self):
        #activation function don't have parameters
        return []
    

class ReLU (Module): 
    def __init__(self): 
        self.input=None
        
    def forward(self,x):
        #save the input for backward 
        self.input=x
        y=(self.input>=0).float()*self.input
        return y
        
    def backward(self, grad ):  
        y=grad
        y[self.input<0]=0
        return y
    
    def param(self):
        #activation function don't have parameters
        return []

    
class MSELoss(Module):
    #MSE loss used to calculate the loss
    def __init__(self): 
        self.v=None
        self.t=None 

    def forward(self,output,target):
        #save parameter output, target and number of points and performing loss calculation
        self.t = target
        self.o = output
        self.n=self.t.size()[1]
        return (self.o - self.t).pow(2).sum()/self.n #diviser par n ou pas?
    
    def backward(self): 
        #performing grad loss calculation 
        grad=2*(self.o-self.t)/self.n
        return grad 
    
    def param(self):
        #Loss function don't have parameters
        return []
    

class Sequential(object): 
    #Sequential class to build Multi layer networks
   
    def __init__(self, *list): 
        #store the leement in a list
        moduleList=[]
        for module in list: 
            moduleList.append(module)
        self.modules=moduleList
        
    def forward(self, x): 
        #go through all the forward of each element
        self.input=x
        for module in self.modules : 
            x=module.forward(x)
        return x 
    
    def backward(self, grad): 
        #go through all the backward of each element
        self.modreverse=reversed(self.modules)
        for module in self.modreverse :
            grad=module.backward(grad)
        return grad
    
    def param(self):
        #store the parameter of each module
        param_mod=[]
        for module in self.modules: 
            l=len(module.param())
            if (l>=1):
                param_mod.append(module) 
            
    
        return param_mod
        
    


