import numpy as np
import torch.cuda
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import math
import matplotlib.pyplot as plt
import CGDs
import pandas as pd
from pyDOE import lhs
from torch import from_numpy
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.DoubleTensor)
print(device)

if device == 'cuda': 
    print(torch.cuda.get_device_name()) 


np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)



path = f"WAN/output/new_WAN_fixed_batch_comb_activation_algo1_no_log"

isDirectory = os.path.isdir(path)

if not os.path.isdir(path):
    os.makedirs(path)
    
if not os.path.isdir(path + "/history"):
    os.makedirs(path + "/history")
    
if not os.path.isdir(path + "/prediction"):
    os.makedirs(path + "/prediction")

if not os.path.isdir(path + "/models"):
    os.makedirs(path + "/models")

# This file generates training data
def trainingData(lb, ub, num_bc, num_f, u, f, RNG_key = None):
  '''
    @param lb: 1d array specifying the lower bound of x and y
    @param ub: 1d array specifying the upper bound of x and y
    @param num_bc: number of points on each side of training region (total number of boundary points = 4 * num_bc)
    @param num_f: number of non-boundary interior points
    @param u: a method that takes in a 2d ndarray as input and returns value of u with given inputs
    @param f: a method that takes in [n * 2]tensors x, y as input and returns value of u_xx+u_yy with given inputs
    
    @return: boundary xy points and inside xy points concatenated, boundary xy points, boundary u values, interior xy points, u_xx+u_yy labels of the interior points
    '''
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
  leftedge_x_y = np.vstack((lb[0] * np.ones(num_bc), lb[1] + (ub[1] - lb[1]) * np.random.rand(num_bc) )).T
  rightedge_x_y = np.vstack((ub[0] * np.ones(num_bc), lb[1] + (ub[1] - lb[1]) * np.random.rand(num_bc) )).T
  topedge_x_y = np.vstack(( lb[0] + (ub[0] - lb[0]) * np.random.rand(num_bc), ub[1] * np.ones(num_bc) )).T
  bottomedge_x_y = np.vstack((lb[0] + (ub[0] - lb[0]) * np.random.rand(num_bc), lb[1] * np.ones(num_bc) )).T
    
  bc_x_y_train = np.vstack([leftedge_x_y, rightedge_x_y, bottomedge_x_y, topedge_x_y]) #x,y pairs on boundaries
  bc_u_train = np.sin(bc_x_y_train[:, 0]) * np.cos(bc_x_y_train[:, 1])
  # bc_u_train = bc_u_train.reshape([-1, 1])
    
  # Latin Hypercube sampling for collocation points
  # num_f sets of tuples(x,t)
  inside_xy = lb + (ub-lb) * lhs(2, num_f)
  # HERE we want code that also generates the training labels (values of f) for the interior points 
  all_xy_train = np.vstack((inside_xy, bc_x_y_train)) # append training points to collocation points


  f_x_y = 2 * np.sin(inside_xy[:, 0]) * np.cos(inside_xy[:, 1])

  f_x_y = torch.from_numpy(f_x_y[:, None]).to(device)
  
  all_xy_train = torch.from_numpy(all_xy_train).to(device)
  bc_x_y_train = torch.from_numpy(bc_x_y_train).to(device)
  bc_u_train = torch.from_numpy(bc_u_train[:, None]).to(device)
  inside_xy = torch.from_numpy(inside_xy).to(device)

#   f_x_y = f(inside_xy[:,[0]], inside_xy[:,[1]])
    
  return all_xy_train, bc_x_y_train, bc_u_train, inside_xy, f_x_y

def testingData(lb, ub, u, f, num):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  X=np.linspace(lb[0], ub[0], num)
  Y=np.linspace(lb[1], ub[1], num)
    
  X, Y = np.meshgrid(X,Y) #X, Y are (256, 256) matrices

  U = u(X,Y)
  u_test = U.flatten('F')[:,None]
  u_test = torch.from_numpy(u_test).float().to(device)
    
  xy_test = np.hstack((X.flatten('F')[:,None], Y.flatten('F')[:,None]))
  f_test = f(xy_test[:,[0]], xy_test[:,[1]])
  f_test = torch.from_numpy(f_test).to(device)

  x_test = torch.from_numpy(xy_test[:,[0]]).to(device)
  y_test = torch.from_numpy(xy_test[:,[1]]).to(device)
#   f_test = f(x_test, y_test)
  return x_test, y_test, xy_test, u_test, f_test, X, Y, U


lb = np.array([-2, -2])
ub = np.array([2, 2])

num_bc = 50
num_f= 5000

u = lambda xy: np.sin(xy[:, 0]) * np.cos([xy[:, 1]]) # this cannot be a torch function, otherwise the gradient would be recorded
f = lambda x, y: 2 * np.sin(x) * np.cos(y) #torch->np
all_xy_train, xy_bc, u_bc, xy_inside, f_xy = trainingData(lb, ub, num_bc, num_f, u, f, 123)

u_test_method = lambda x, y: np.sin(x) * np.cos(y) #takes 2 inputs, but should return same values as previous u
x_test, y_test, xy_test, u_test, f_test, X, Y, U = testingData(lb, ub, u_test_method, f, 256)


class PINN_Poisson_2d(nn.Module):
    def __init__(self, layers, x_test, y_test, u_test, f_test, x_bc, y_bc, u_bc, fxy, x_inside_train, y_inside_train, f_x = None, f_y = None):
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
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        # self.sinc = nn.sinc()
        self.elu = nn.ELU()

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
        # self.x_bc.requires_grad = True
        # self.y_bc.requires_grad = True

        self.xy_bc = torch.cat((self.x_bc, self.y_bc), dim = 1)
        # self.xy_bc.requires_grad = True

        self.u_bc = u_bc
        self.x_inside_train = x_inside_train #for interior poitns PDE training
        self.y_inside_train = y_inside_train
        
        self.xy_interior = torch.cat((self.x_inside_train, self.y_inside_train), dim = 1)
        
        self.x_inside_train.requires_grad = True
        self.y_inside_train.requires_grad = True

        self.fxy = fxy #u_xx+u_yy
        self.f_x = f_x
        self.f_y = f_y
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)

        # std = gain * sqrt(2/(input_dim+output_dim))
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)

        'foward pass'
    def forward(self, x, y):
        _x = x.clone()
        _y = y.clone()
        a = torch.concat((_x, _y), dim = 1)
        # print(a.shape)

        for i in range(len(self.linears) - 1):
            
            z = self.linears[i](a)
                        
            # a = self.tanh(z)

            if i == 0: a = self.softplus(z)
            elif i % 2 == 1: a = self.softplus(z)
            else: a = torch.sin(z)
            
        a = self.linears[-1](a)
        
        return a

    def grad_u(self, x, y):
        # _x = x.clone()
        # _y = y.clone()
        

        # if _x.requires_grad == False:
        #   _x.requires_grad = True
          
          
        # if _y.requires_grad == False:
        #   _y.requires_grad = True
        
        
        u_val = self.forward(x, y)

        grad_u_x = autograd.grad(u_val, x, grad_outputs = torch.zeros_like(x), create_graph=True, retain_graph = True)[0]
        grad_u_y = autograd.grad(u_val, y, grad_outputs = torch.zeros_like(y), create_graph=True, retain_graph = True)[0]
        
        grad_u_xx = autograd.grad(grad_u_x, x, grad_outputs = torch.zeros_like(x), create_graph=True, retain_graph = True)[0]
        grad_u_yy = autograd.grad(grad_u_y, y, grad_outputs = torch.zeros_like(y), create_graph=True, retain_graph = True)[0]
        
        grad_u = torch.cat((grad_u_x, grad_u_y), dim = 1)
        laplacian_u = torch.cat((grad_u_xx, grad_u_yy), dim = 1)
        return u_val, grad_u, laplacian_u



    def loss_BC(self, x_bc = None, y_bc = None, u_bc = None):
        '''
        the parameters must be all None or all non-None 
        '''
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
        _x = self.x_inside_train.clone()
        _y = self.y_inside_train.clone()
        
        u = self.forward(_x, _y)

        u_x = autograd.grad(u,_x,torch.ones([_x.shape[0], 1]).to(device), retain_graph= True, create_graph=True)[0]
        u_y = autograd.grad(u,_y,torch.ones([_y.shape[0], 1]).to(device),retain_graph= True, create_graph=True)[0]

        u_xx = autograd.grad(u_x,_x,torch.ones((_x.shape[0], 1)).to(device), create_graph = True)[0]
        u_yy = autograd.grad(u_y,_y,torch.ones((_y.shape[0], 1)).to(device), create_graph = True)[0]

        shouldBeZero = u_xx + u_yy + self.fxy
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
    def __init__(self, layers):
        super(Discriminator, self).__init__()
        self.layers = layers
        self.linears = nn.ModuleList(
            [nn.Linear(layers[0], layers[2])]
             + [nn.Linear(layers[2], layers[2])] * (layers[1])+ [nn.Linear(layers[2], layers[3])]
        )
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x, y):
        temp = torch.cat((x, y), dim = 1)
        a = temp.clone()
        for i in range(len(self.linears) - 1):
            # print(i)
            a = self.linears[i](a)
            # temp = self.relu(temp)

            if i == 0: a = self.tanh(a)
            elif i % 2 == 0: a = self.softplus(a)
            else: a = torch.sin(a)
            
        a = self.linears[-1](a)

        return a


    def grad_v(self, x, y):
        # _x = x.clone()
        # _y = y.clone()
        

        # if _x.requires_grad == False:
        #   _x.requires_grad = True
          
          
        # if _y.requires_grad == False:
        #   _y.requires_grad = True
        
        v_val = self.forward(x, y)

        
        grad_v_x = autograd.grad(v_val, x, grad_outputs = torch.zeros_like(x), create_graph=True, retain_graph = True)[0]
        grad_v_y = autograd.grad(v_val, y, grad_outputs = torch.zeros_like(y), create_graph=True, retain_graph = True)[0]
        
        grad_v = torch.cat((grad_v_x, grad_v_y), dim = 1)
        return v_val, grad_v



layers = [2, 40, 40, 40, 40, 40, 40, 1]
# beta_int, beta_intw, beta_bd= 1.0, 1.0, 1.0

u = lambda xy: np.sin(xy[:, 0]) * np.cos([xy[:, 1]]) # this cannot be a torch function, otherwise the gradient would be recorded
f = lambda x, y: -2 * np.sin(x) * np.cos(y) #torch->np
all_xy_train, xy_bc, u_bc, xy_inside, f_xy = trainingData(lb, ub, num_bc, num_f, u, f, 123)

u_test_method = lambda x, y: np.sin(x) * np.cos(y) #takes 2 inputs, but should return same values as previous u
x_test, y_test, xy_test, u_test, f_test, X, Y, U = testingData(lb, ub, u_test_method, f, 256)



PINN = PINN_Poisson_2d(layers, x_test, y_test, u_test, f_test,
                                xy_bc[:,[0]], xy_bc[:,[1]], u_bc, 
                                f_xy, xy_inside[:,[0]], xy_inside[:,[1]]).to(device)
discriminator = Discriminator([2, 6, 40, 1]).to(device)





PINN.to(device)
discriminator.to(device)

print(PINN)
print(discriminator)

optimizer = CGDs.GACGD(x_params = discriminator.parameters(), y_params = PINN.parameters(), lr_x = 0.001, lr_y = 0.001, eps = 1e-8, max_iter = 500,
                       tol = 1e-7, atol = 1e-20, track_cond = lambda x, y: True)


# AdamOptimizer = torch.optim.Adam(params = PINN.parameters(), lr = 0.001)
# AdaGradOptimizer = torch.optim.Adagrad(params = discriminator.parameters(), lr = 0.015)
max_iter = 250001
recordPer = 100
graphPer = 0
iter_num_sum = 0
savePer = 5000
WAN_algo1_Info = pd.DataFrame([])
dummy = torch.tensor([[0.0]]).to(device)

# PINN.x_bc.requires_grad = True
# PINN.y_bc.requires_grad = True

for i in range(max_iter):
  optimizer.zero_grad()
  # AdamOptimizer.zero_grad()
  
  # g_loss, PINN_loss_bc, g_pde_loss = PINN.loss()
  # g_loss.backward()
  # AdamOptimizer.step()

  pdeDiff = PINN.test_PDE()
  penalty = discriminator(PINN.x_inside_train, PINN.y_inside_train)

  A_norm = torch.norm(pdeDiff * penalty, 2)
  operator_norm = torch.norm(penalty, 2)

  loss_int = (A_norm ** 2) / (operator_norm ** 2)

  loss_BC = PINN.loss_BC()

  WAN_loss = loss_int + loss_BC

  # WAN_loss.backward(retain_graph = True)
  loss_max = -WAN_loss.clone()

  optimizer.step(loss_x = loss_max, loss_y = WAN_loss, trigger = 0)
  iter_num_sum += optimizer.info["num_iter"]
  if i % recordPer == 0:
    
      g_loss, PINN_loss_bc, g_pde_loss = PINN.loss()
      error_vec, u_pred = PINN.test(graphPer != 0 and i % graphPer == 0)
      
      WAN_algo1_Info = WAN_algo1_Info.append({
        "iter": i,
        "L2 error": error_vec,
        "PINN loss": g_loss.item(),
        "PINN BC loss": PINN_loss_bc.item(),
        "WAN loss": WAN_loss.item(),
        "loss_int": loss_int.item(),
        "A_norm": A_norm.item(),
        "operator_norm": operator_norm.item(),
        "iter_num_sum" : iter_num_sum
        }, ignore_index = True)
      # print(w_val)
  if i % savePer == 0:
    WAN_algo1_Info.to_csv(f"{path}/history/iter_{i}.csv")
    np.savetxt(f"{path}/prediction/PINNPrediction_iter_{i}.csv", u_pred)
    torch.save({
            "PINN_state_dict": PINN.state_dict(),
            "Discriminator_state_dict": discriminator.state_dict(),
            "GACGD_optimizer_state_dict" : optimizer.state_dict(),
        }, f"{path}/models/models_iter_{i}.pt")

