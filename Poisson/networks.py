# This is supposed to be a template for how to implement PINNS models for the code
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import math
import matplotlib.pyplot as plt


# This class represents a PINN solving a Poisson problem in 2 dimensions
class PINN_Poisson_2d(nn.Module):
    def __init__(self, layers, x_test, y_test, u_test, f_test, x_bc, y_bc, u_bc, fxy, x_inside_train, y_inside_train, RNG_key = None):
        '''
        @param layers: number of input/output neurons in each layer of the model
        @param x_test: n*1 tensor, used for testing
        @param y_test: n*1 tensor, used for testing
        @param u_test: n*1 tensor, used for testing
        @param f_test: n*1 tensor, used for testing
        
        @param x_bc: n*1 tensor, used in determining boundary losses
        @param y_bc: n*1 tensor, used in determining boundary losses
        @param u_bc: n*1 tensor, used in determining boundary losses
        
        @param fxy: n*1 tensor, used in determining interior points' PDE losses in training
        @param x_inside_train: n*1 tensor, used in determining interior points' PDE losses
        @param y_inside_train: n*1 tensor, used in determining interior points' PDE losses
        
        '''
        super().__init__() #call __init__ from parent class 
              
        'activation function'
        self.activation = nn.Tanh()

        'loss function as MSE'
        self.loss_function = nn.MSELoss(reduction ='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        
        self.iter = 0 
        self.layers = layers
        
        self.x_test = x_test #used for testing
        self.y_test = y_test
        self.u_test = u_test
        self.f_test = f_test
        
        self.x_bc = x_bc #boundaries
        self.y_bc = y_bc
        self.u_bc = u_bc
        self.x_inside_train = x_inside_train #for interior poitns PDE training
        self.y_inside_train = y_inside_train
        self.fxy = fxy #u_xx+u_yy
        
        if RNG_key == None:
            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
        else:
            torch.manual_seed(RNG_key)
            torch.cuda.manual_seed_all(RNG_key)

        # std = gain * sqrt(2/(input_dim+output_dim))
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)

        'foward pass'
    def forward(self, x, y):
        
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x).to(device)
        if torch.is_tensor(y) != True:         
            y = torch.from_numpy(y).to(device)
        
        a = torch.cat((x, y),1)
        # print(a.shape)

        for i in range(len(self.layers)-2):
            
            z = self.linears[i](a)
                        
            a = self.activation(z)
            
        a = self.linears[-1](a)
        
        return a

    def loss_BC(self, x_bc = None, y_bc = None, u_bc = None):
        '''
        the parameters must be all None or all non-None 
        '''
        if x_bc != None:
            return self.loss_function(self.forward(x_bc, y_bc), u_bc)
        else:
            return self.loss_function(self.forward(self.x_bc, self.y_bc), self.u_bc)
    
    # We do not want to hard code the right hand side. Instead, we want to pass it as a label, just like the boundary conditions
    def loss_PDE(self, x = None, y = None, fxy = None):
        '''
        the parameters must be all None or all non-None 
        '''
        shouldBeZero = self.test_PDE(self.x_inside_train, self.y_inside_train, self.fxy)
        
        if x !=  None:
            shouldBeZero = self.test_PDE(x, y, fxy)

        loss_f = self.loss_function(shouldBeZero, torch.zeros_like(shouldBeZero))
                
        return loss_f

    def test_PDE(self, x = None, y = None, fxy = None):
        '''
        the parameters must be all None or all non-None 
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _x = self.x_inside_train.clone()
        _y = self.y_inside_train.clone()
        _fxy = self.fxy.clone()

        if x != None:
            _x = x.clone()
            _y = y.clone()
            _fxy = fxy.clone()
            
        _x.requires_grad = True
        _y.requires_grad = True
        
        u = self.forward(_x, _y)

        u_x = autograd.grad(u,_x,torch.ones([_x.shape[0], 1]).to(device), retain_graph= True, create_graph=True)[0]
        u_y = autograd.grad(u,_y,torch.ones([_y.shape[0], 1]).to(device),retain_graph= True, create_graph=True)[0]

        u_xx = autograd.grad(u_x,_x,torch.ones((_x.shape[0], 1)).to(device), create_graph = True)[0]
        u_yy = autograd.grad(u_y,_y,torch.ones((_y.shape[0], 1)).to(device), create_graph = True)[0]

        shouldBeZero = u_xx + u_yy - _fxy
        shouldBeZero.retain_graph = True
        
        return shouldBeZero

    def loss(self, x_bc = None, y_bc = None, u_bc = None, x_inside=None, y_inside=None, fxy=None):
        '''
            if no resampling is needed in training iterations, do not provide values for the parameters
        
            the first 3 parameters must be all None or all non-None 
            
            the last 3 parameters must be all None or all non-None 
        
        '''
        loss_u = 0
        loss_f = 0
        if x_bc != None:
            loss_u = self.loss_BC(x_bc, y_bc, u_bc)
        else:
            loss_u = self.loss_BC()
            
        if x_inside != None:
            loss_f = self.loss_PDE(x_inside, y_inside, fxy)
        else:
            loss_f = self.loss_PDE()
        
        loss_val = loss_u + loss_f

        return loss_val, loss_u, loss_f
         
    # I am not sure this function should be part of the neural network, and if it is, it should again work with a general f
    # 'test neural network'
    def test(self, graph = False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _x = self.x_test.clone()
        _y = self.y_test.clone()
        _x.requires_grad = True
        _y.requires_grad = True
        
        u_pred = self.forward(_x, _y)
        
        u_x = autograd.grad(u_pred,_x,torch.ones([_x.shape[0], 1]).to(device), retain_graph= True, create_graph=True)[0]      
        u_y = autograd.grad(u_pred,_y,torch.ones([_y.shape[0], 1]).to(device), retain_graph= True, create_graph=True)[0]

        u_xx = autograd.grad(u_x, _x, torch.ones((_x.shape[0], 1)).to(device) , retain_graph = True)[0]
        u_yy = autograd.grad(u_y, _y, torch.ones((_y.shape[0], 1)).to(device) , retain_graph = True)[0]

        fxy_test = self.f_test
        
        PDE = u_xx + u_yy - fxy_test
        
        test_size = int(math.sqrt(self.x_test.shape[0]))
        
        PDE = PDE.cpu().detach().numpy()
        
        PDE = np.reshape(PDE, (test_size, test_size), order='F') #PDE is a 256*256 matrix now

        u_pred = u_pred.cpu().detach().numpy()
        
        error_vec = np.linalg.norm((self.u_test.cpu().detach().numpy() - u_pred),2) / np.linalg.norm(self.u_test.cpu().detach().numpy(), 2)  # Relative L2 Norm of the error (Vector)
        
        u_pred = np.reshape(u_pred, (test_size, test_size), order='F')
        u_test_graph = np.reshape(self.u_test.cpu().detach().numpy(),(test_size, test_size), order='F') # make u_test a matrix for graphing
        if (graph):
          fig, ax = plt.subplots(1, 4,figsize=(30,4))
          
          im1 = ax[0].imshow(u_test_graph, interpolation='nearest', cmap='rainbow', 
#                       extent=[self.y_test.min(), self.y_test.max(), self.x_test.min(), self.x_test.max()], 
                      origin='lower', aspect='equal')
          fig.colorbar(im1, ax = ax[0])
          
          im2 = ax[1].imshow(u_pred, interpolation='nearest', cmap='rainbow', 
#                       extent=[self.y_test.min(), self.y_test.max(), self.x_test.min(), self.x_test.max()], 
                      origin='lower', aspect='equal')
          fig.colorbar(im2, ax = ax[1])
          
          im3 = ax[2].imshow(u_pred - u_test_graph, interpolation='nearest', cmap='bwr', 
#                       extent=[self.y_test.min(), self.y_test.max(), self.x_test.min(), self.x_test.max()], 
                      origin='lower', aspect='equal')
          fig.colorbar(im3, ax=ax[2])
          
          im4 = ax[3].imshow(PDE, interpolation='nearest', cmap='bwr', 
#                       extent=[self.y_test.min(), self.y_test.max(), self.x_test.min(), self.x_test.max()], 
                      origin='lower', aspect='equal')


          ax[0].set_title('True $u(x,y)$', fontsize = 10)
          ax[1].set_title('Predicted $u(x,y)$', fontsize = 10)
          ax[2].set_title('Difference', fontsize = 10)
          ax[3].set_title('PDE diff', fontsize = 10)

          fig.colorbar(im4, ax = ax[3])
          plt.show()
        return error_vec, u_pred

# The Discriminator Network
# A general discriminator. Input_size is the physical dimension of the problem, output_size the dimension of the residual
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map = nn.Sequential(
            nn.Linear(input_size, 2 * hidden_size),
            nn.ReLU(),
#             nn.Tanh(),
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.ReLU(),
#             nn.Tanh(),
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.ReLU(),
#             nn.Tanh(),
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.ReLU(),
#             nn.Tanh(),
            nn.Linear(2  * hidden_size, output_size),
        )

    def forward(self, x, y):
        return self.map(torch.cat((x,y), 1))