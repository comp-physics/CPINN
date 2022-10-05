import numpy as np
import torch
import time
import pandas as pd
import scipy.io
from pyDOE import lhs
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
np.random.seed(123)
q = 1
layers = [2, 200, 200, 200, 200, 1]
lb = np.array([-1.0])
ub = np.array([1.0])

N = 200

data = scipy.io.loadmat('AC/AC.mat')

t = data['tt'].flatten()[:,None] # T x 1
x = data['x'].flatten()[:,None] # N x 1
Exact = np.real(data['uu']).T # T x N

TT, XX = np.meshgrid(data["tt"][0], data["x"][0])

# load the data
data = scipy.io.loadmat('AC/AC.mat')
usol = data['uu']

N_f = 10000
N_u = 200

# Grid
t = data['tt'][0]
x = data['x'][0]


X, T = np.meshgrid(x,t)

x_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.flatten()[:,None]              

np.random.seed(123)

def gatherCollocationData(lb, ub, size):
  

  x = lb + (ub-lb) * lhs(2, size) # for PDE constraint

  return x

lbs = [np.array([-1, 0.1 * i]) for i in range(10)]
ubs = [np.array([1, 0.1 * (i + 1)]) for i in range(10)]


points = []

for i in range(len(lbs)):
    x = gatherCollocationData(lbs[i], ubs[i], 200)
    points.append(x)



import torch.nn as nn
import torch.autograd as autograd

class PINN_AC(nn.Module):
    def __init__(self, x_test, t_test, u_test, N_ic, N_bc, N_f, layers):
        super().__init__() #call __init__ from parent class 
              
        'activation function'
        self.activation = nn.Tanh()

        'loss function as MSE'
        self.loss_function = nn.MSELoss(reduction = 'mean')
        self.hiddenLayers = [nn.Linear(layers[0], layers[2])] + [nn.Linear(layers[2], layers[2]) for i in range(layers[1] - 1)] + [nn.Linear(layers[2], layers[3])]

        self.linears = nn.ModuleList(self.hiddenLayers)
        
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)

        # std = gain * sqrt(2/(input_dim+output_dim))
        for i in range(len(self.linears)):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)

        self.iter = 0 
        
        
        self.x_ic = np.random.rand(N_ic, 1) * 2 - 1
        self.x_ic_tensor = torch.from_numpy(self.x_ic).to(device)
        self.t_ic_tensor = torch.zeros_like(self.x_ic_tensor)

        self.u_ic = torch.from_numpy(self.x_ic ** 2 * np.cos(np.pi * self.x_ic)).to(device)
        self.t_bc = torch.from_numpy(np.random.rand(N_bc, 1)).to(device)

        lb = np.array([-1, 0])
        ub = np.array([1, 1])

        self.xt_train = lb + (ub - lb)*lhs(2, N_f) # for PDE constraint
        # print(self.xt_train.shape)

        
        ones = np.ones([self.t_bc.shape[0], 1])
        self.ones_tensor = torch.from_numpy(ones).to(device)
        self.ones_tensor.requires_grad = True

        self.n_ones_tensor = torch.from_numpy(ones * (-1)).to(device)
        self.n_ones_tensor.requires_grad = True


        self.x_train = torch.from_numpy(self.xt_train[:, 0])[:, None].to(device)
        self.t_train = torch.from_numpy(self.xt_train[:, 1])[:, None].to(device)

        self.x_train.requires_grad = True
        self.t_train.requires_grad = True
        
        self.lb = torch.from_numpy(lb).to(device)
        self.ub = torch.from_numpy(ub).to(device)

        self.x_test = torch.from_numpy(x_test[:, None]).to(device)
        self.t_test = torch.from_numpy(t_test[:, None]).to(device)
        self.u_test = u_test
        self.u_test_graph = u_test.reshape(512, 201, order = "F")
        'foward pass'
    def forward(self, x, t):
        
        # print("forward() x shape", x.shape)
        # print("forward() t shape", t.shape)


        a = torch.cat((x, t), 1)
        
        # a = 2.0 * (a - self.lb)/(self.ub - self.lb) - 1.0

        # print(a.shape)

        for i in range(len(self.linears)-1):
            
            z = self.linears[i](a)
                        
            a = self.activation(z)
            
        a = self.linears[-1](a)
        
        return a


    def loss_IC(self):
        out = self.forward(self.x_ic_tensor, self.t_ic_tensor)
        

        return self.loss_function(out, self.u_ic), out - self.u_ic
    def loss_BC(self):
        '''
        the parameters must be all None or all non-None 
        '''
        _u_output1 = self.forward(self.ones_tensor, self.t_bc)
        _u_output2 = self.forward(self.n_ones_tensor, self.t_bc)


        loss_bc_u = self.loss_function(_u_output1, _u_output2)
        
        _u_output1_x = autograd.grad(_u_output1, self.ones_tensor, torch.ones([self.ones_tensor.shape[0], 1]).to(device), create_graph=True, retain_graph=True)[0]

        
        _u_output2_x = autograd.grad(_u_output2, self.n_ones_tensor, torch.ones([self.n_ones_tensor.shape[0], 1]).to(device), create_graph=True, retain_graph=True)[0]

        loss_bc_u_x = self.loss_function(_u_output1_x, _u_output2_x)
        return loss_bc_u + loss_bc_u_x, loss_bc_u, loss_bc_u_x, _u_output1 - _u_output2, _u_output1_x - _u_output2_x
    
    # We do not want to hard code the right hand side. Instead, we want to pass it as a label, just like the boundary conditions
    def loss_PDE(self, xt_collocation):
        '''
        the parameters must be all None or all non-None 
        '''


        pdeDiff = self.PDE_residual(xt_collocation)
        
        loss_f = self.loss_function(pdeDiff, torch.zeros_like(pdeDiff))
                
        return loss_f, pdeDiff

    def PDE_residual(self, xt_collocation):
        '''
        the parameters must be all None or all non-None 
        '''
        if ((xt_collocation != None).all()):


          xt_collocation = torch.from_numpy(xt_collocation).to(device)
          
          x_collocation = xt_collocation[:, [0]]
          
          t_collocation = xt_collocation[:, [1]]

          x_collocation.requires_grad = True
          t_collocation.requires_grad = True
          
          
          u = self.forward(x_collocation, t_collocation)


          u_x = autograd.grad(u, x_collocation, torch.ones([x_collocation.shape[0], 1]).to(device), retain_graph= True, create_graph=True)[0]
          u_t = autograd.grad(u, t_collocation, torch.ones([t_collocation.shape[0], 1]).to(device), retain_graph= True, create_graph=True)[0]

          u_xx = autograd.grad(u_x, x_collocation, torch.ones((x_collocation.shape[0], 1)).to(device), retain_graph= True, create_graph = True)[0]

          shouldBeZero = u_t - 0.0001 * u_xx + 5 * (u ** 3) - 5 * u
          shouldBeZero.retain_graph = True
          
          return shouldBeZero

        else:

          
          u = self.forward(self.x_train, self.t_train)

          u_x = autograd.grad(u, self.x_train,torch.ones([self.x_train.shape[0], 1]).to(device), retain_graph= True, create_graph=True)[0]
          u_t = autograd.grad(u, self.t_train,torch.ones([self.t_train.shape[0], 1]).to(device), retain_graph= True, create_graph=True)[0]

          u_xx = autograd.grad(u_x, self.x_train,torch.ones((self.x_train.shape[0], 1)).to(device), retain_graph= True, create_graph = True)[0]

          shouldBeZero = u_t - 0.0001 * u_xx + 5 * (u ** 3) - 5 * u
          shouldBeZero.retain_graph = True
          
          return shouldBeZero
          

    def loss(self, xt_collocation = None):
        '''
            if no resampling is needed in training iterations, do not provide values for the parameters
        
            the first 3 parameters must be all None or all non-None 
            
            the last 3 parameters must be all None or all non-None 
        
        '''
        
        loss_BC, _, _, bc_diff1, bc_diff2 = self.loss_BC()
            
        loss_PDE, _ = self.loss_PDE(xt_collocation)

        loss_IC, ic_diff = self.loss_IC()
        
        loss_val = loss_PDE + loss_IC + loss_BC

        return loss_val, loss_PDE, loss_IC, loss_BC
         
    # I am not sure this function should be part of the neural network, and if it is, it should again work with a general f
    # 'test neural network'
    def test(self, graph = False):

        # print(self.x_test.shape)
        u_pred = self.forward(self.x_test, self.t_test)
        
        # u_x = autograd.grad(u_pred,_x,torch.ones([_x.shape[0], 1]).to(device), retain_graph= True, create_graph=True)[0]      
        # u_y = autograd.grad(u_pred,_y,torch.ones([_y.shape[0], 1]).to(device), retain_graph= True, create_graph=True)[0]

        # u_xx = autograd.grad(u_x, _x, torch.ones((_x.shape[0], 1)).to(device) , retain_graph = True)[0]
        # u_yy = autograd.grad(u_y, _y, torch.ones((_y.shape[0], 1)).to(device) , retain_graph = True)[0]

        # fxy_test = self.f_test
        
        # PDE = u_xx + u_yy - fxy_test
        
        # test_size = int(math.sqrt(self.x_test.shape[0]))
        
        # PDE = PDE.cpu().detach().numpy()
        
        # PDE = np.reshape(PDE, (test_size, test_size), order='F') #PDE is a 256*256 matrix now

        u_pred = u_pred.cpu().detach().numpy()
        
        # print(u_pred.shape)
        # print(self.u_test.shape)
        error_vec = np.linalg.norm((self.u_test - u_pred), 2) / np.linalg.norm(self.u_test, 2)  # Relative L2 Norm of the error (Vector)
        
        u_pred_graph = u_pred.reshape(512, 201, order='F')
        
        if (graph):
          fig, ax = plt.subplots(1, 3, figsize=(22,4))

          ax[0].set_title("u_pred")
          h = ax[0].imshow(u_pred_graph, interpolation='nearest', cmap='jet', 
                        extent=[t.min(), t.max(), x.min(), x.max()], 

                        origin='lower', aspect='auto')
          divider = make_axes_locatable(ax[0])
          cax = divider.append_axes("right", size="5%", pad=0.05)
          fig.colorbar(h, cax=cax)

# true difference plot
          ax[1].set_title("u_true")
          h = ax[1].imshow(self.u_test_graph, interpolation='nearest', cmap='jet', 
                        extent=[t.min(), t.max(), x.min(), x.max()], 
                        origin='lower', aspect='auto')
          divider = make_axes_locatable(ax[1])
          cax = divider.append_axes("right", size="5%", pad=0.05)
          fig.colorbar(h, cax=cax)

          ax[2].set_title("u_pred - u_true")
          # h = ax[2].imshow(np.log(np.abs(u_pred.T - Exact.T)), interpolation='nearest', cmap='rainbow', 
          h = ax[2].imshow(u_pred_graph - self.u_test_graph, interpolation='nearest', cmap='jet',
                        extent=[t.min(), t.max(), x.min(), x.max()], 
                        origin='lower', aspect='auto')
          divider = make_axes_locatable(ax[2])
          cax = divider.append_axes("right", size="5%", pad=0.05)


          fig.colorbar(h, cax=cax)
          plt.show()
        return error_vec, u_pred

N_ic = 100
N_bc = 256
N_f = 10000

layers = [2, 4, 128, 1]

import torch.optim as optim
import time
max_iter = 1400001

lr = 0.001
betas = (0.99, 0.99)

PINN_Adam = PINN_AC(x_star[:, 0], x_star[:, 1], u_star, N_ic, N_bc, N_f, layers).to(device).double()
print(PINN_Adam)

optimizer = optim.Adam(PINN_Adam.parameters(), lr=lr,betas=betas, eps=1e-08, weight_decay=0, amsgrad=False)

import argparse
parser = argparse.ArgumentParser(description='Enter the parameters')
parser.add_argument('-adaptiveTol','--adaptiveTol', help='adaptive tolerance for subinterval', type = float, required=True)

args = parser.parse_args()

adaptiveTol = args.adaptiveTol

cols = ["iter", "L2 Error", "PINN loss", "PINN BC loss", "PINN IC loss", "PINN PDE loss"]
AdamInfo = pd.DataFrame(columns = cols)
print(AdamInfo.head())


path = f"AC/output/Adaptive_Adam_adaptiveTol{adaptiveTol}"

isDirectory = os.path.isdir(path)

if not os.path.isdir(path):
    os.makedirs(path)
    
if not os.path.isdir(path + "/history"):
    os.makedirs(path + "/history")
    
if not os.path.isdir(path + "/prediction"):
    os.makedirs(path + "/prediction")

if not os.path.isdir(path + "/models"):
    os.makedirs(path + "/models")


recordPer = 500
graphPer = 0
savePer = 100000
# Info = []

interval = 0
size = 200


x = points[interval]
switchAt = []

start_time = time.time()
for i in range(max_iter):
    optimizer.zero_grad()
    

    loss, loss_PDE, loss_IC, loss_BC = PINN_Adam.loss(x)

    if (loss_PDE.item() < adaptiveTol and interval != 9):
      
      interval += 1
      print("Next interval!")
      switchAt.append([interval, i])
      np.savetxt(f"{path}/history/Interval_Info.csv", switchAt)
      x = np.vstack((x, points[interval]))
    #   iterRemaining = 2000

    
    loss.backward()
    optimizer.step()

    if i % recordPer == 0:
        error_vec, u_pred = PINN_Adam.test(graphPer != 0 and i % graphPer == 0)
        new_row = {"iter":i, 
              "L2 Error": error_vec, 
              "PINN loss": loss.item(), 
              "PINN BC loss": loss_BC.item(), 
              "PINN IC loss": loss_IC.item(), 
              "PINN PDE loss": loss_PDE.item()
              }
        AdamInfo.loc[len(AdamInfo)] = new_row
        # AdamInfo = pd.concat([AdamInfo, pd.DataFrame(new_row.values(), columns=AdamInfo.columns)], ignore_index = True)
        # print(f"iter: {i}, L2 relative error: {error_vec}, PINN loss: {loss}, loss_pde: {loss_PDE.item()}, loss_bc: {loss_BC.item()}, loss_ic: {loss_IC.item()}")

    if i % savePer == 0:
        AdamInfo.to_csv(f"{path}/history/Adam_history_iter_{i}.csv")
        np.savetxt(path + f"/prediction/u_pred_iter_{i}.csv", u_pred)
        torch.save({"PINN_state": PINN_Adam.state_dict(),
            "Adam_state" : optimizer.state_dict()}, path + f"/models/PINNAdam_model_iter_{i}.pt")

print("Training Time: ", time.time() - start_time)