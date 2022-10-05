import numpy as np
import torch
import time
import pandas as pd
import scipy.io
from pyDOE import lhs
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.DoubleTensor)

np.random.seed(1234)
import matplotlib.pyplot as plt

# layers = [2, 200, 200, 200, 200, 1]
lb = np.array([-1.0])
ub = np.array([1.0])

N = 200

data = scipy.io.loadmat('AC/AC.mat')

t = data['tt'].flatten()[:,None] # T x 1
x = data['x'].flatten()[:,None] # N x 1
Exact = np.real(data['uu']).T # T x N

TT, XX = np.meshgrid(data["tt"][0], data["x"][0])


usol = data['uu']

N_f = 10000
N_u = 200

# Grid
t = data['tt'][0]
x = data['x'][0]


X, T = np.meshgrid(x,t)

x_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.flatten()[:,None]              


idx = np.random.choice(x_star.shape[0], N_u, replace=False)
X_train = x_star[idx, :]
u_train = u_star[idx,: ]



import torch.nn as nn
import torch.autograd as autograd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(1234)

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
        if (xt_collocation is not None):


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
        
        u_pred = u_pred.cpu().detach().numpy()
        
        error_vec = np.linalg.norm((self.u_test - u_pred), 2) / np.linalg.norm(self.u_test, 2)  # Relative L2 Norm of the error (Vector)
        
        # u_pred_graph = u_pred.reshape(512, 201, order='F')
        
#         if (graph):
#           fig, ax = plt.subplots(1, 3, figsize=(22,4))

#           ax[0].set_title("u_pred")
#           h = ax[0].imshow(u_pred_graph, interpolation='nearest', cmap='jet', 
#                         extent=[t.min(), t.max(), x.min(), x.max()], 

#                         origin='lower', aspect='auto')
#           divider = make_axes_locatable(ax[0])
#           cax = divider.append_axes("right", size="5%", pad=0.05)
#           fig.colorbar(h, cax=cax)

# # true difference plot
#           ax[1].set_title("u_true")
#           h = ax[1].imshow(self.u_test_graph, interpolation='nearest', cmap='jet', 
#                         extent=[t.min(), t.max(), x.min(), x.max()], 
#                         origin='lower', aspect='auto')
#           divider = make_axes_locatable(ax[1])
#           cax = divider.append_axes("right", size="5%", pad=0.05)
#           fig.colorbar(h, cax=cax)

#           ax[2].set_title("u_pred - u_true")
#           # h = ax[2].imshow(np.log(np.abs(u_pred.T - Exact.T)), interpolation='nearest', cmap='rainbow', 
#           h = ax[2].imshow(u_pred_graph - self.u_test_graph, interpolation='nearest', cmap='jet',
#                         extent=[t.min(), t.max(), x.min(), x.max()], 
#                         origin='lower', aspect='auto')
#           divider = make_axes_locatable(ax[2])
#           cax = divider.append_axes("right", size="5%", pad=0.05)


#           fig.colorbar(h, cax=cax)
#           plt.show()
        return error_vec, u_pred


class Discriminator(nn.Module):
    def __init__(self, layers):
        super(Discriminator, self).__init__()
        self.hiddenLayers = [nn.Linear(layers[0], layers[2])] + [nn.ReLU(), nn.Linear(layers[2], layers[2])] * (layers[1] - 1)+ [nn.ReLU(), nn.Linear(layers[2], layers[3])]
        self.linears = nn.ModuleList(self.hiddenLayers)

    def forward(self, x, y):
        temp = torch.cat((x,y), 1)
        for layer in self.linears:
            temp = layer(temp)
        return temp


N_ic = 100
N_bc = 256
N_f = 10000



import time
import CGDs
import itertools

import argparse
parser = argparse.ArgumentParser(description='Enter the parameters')
parser.add_argument('-PINN_neurons','--PINN_neurons', help='number of neurons of PINN per layer', type = int, required=True)
parser.add_argument('-dis_neurons','--dis_neurons', help='number of neurons of discriminator per layer', type = int, required=True)

args = parser.parse_args()

PINN_neurons = args.PINN_neurons
dis_neurons = args.dis_neurons
# dis_layers = [2, 4, 128, 4]
dis_layers = [2, 4, dis_neurons , 4]
discriminator = Discriminator(dis_layers).to(device).double()

print(discriminator)
max_iter = 13251

adaptiveTol = 1e-7

betas = (0.99, 0.99)

layers = [2, 4, PINN_neurons, 1]
lr_max = 0.001
lr_min = 0.001


PINN_GACGD = PINN_AC(x_star[:, 0], x_star[:, 1], u_star, N_ic, N_bc, N_f, layers).to(device).double()
print(PINN_GACGD)


path = f"AC/output/new_adaptive_{adaptiveTol}_1dis_GACGD_lrmax_{lr_max}_lrmin_{lr_min}_PINN_{layers[1]}_{layers[2]}_dis_{dis_layers[1]}_{dis_layers[2]}"

isDirectory = os.path.isdir(path)

if not os.path.isdir(path):
    os.makedirs(path)
    
if not os.path.isdir(path + "/history"):
    os.makedirs(path + "/history")
    
if not os.path.isdir(path + "/prediction"):
    os.makedirs(path + "/prediction")

if not os.path.isdir(path + "/models"):
    os.makedirs(path + "/models")

GACGD_optimizer = CGDs.GACGD(x_params = discriminator.parameters(), 
                           y_params = PINN_GACGD.parameters(), max_iter = 500,
            lr_x=lr_max, lr_y=lr_min, tol=1e-7, atol = 1e-20, eps=1e-8, beta=0.99, track_cond = lambda x, y: True)
print(GACGD_optimizer.state_dict())
# cols = ["iter", "L2 Error", "PINN loss", "CPINN loss", "PINN BC loss", "PINN IC loss", "PINN PDE loss", "iter_num_sum"]
ACGDInfo = pd.DataFrame()
print(ACGDInfo.head())

recordPer = 50
graphPer = 0
savePer = 250



def gatherCollocationData(lb, ub, size):
  x = lb + (ub-lb) * lhs(2, size) # for PDE constraint

  return x

lbs = [np.array([-1, 0.1 * i]) for i in range(10)]
ubs = [np.array([1, 0.1 * (i + 1)]) for i in range(10)]


points = []

for i in range(len(lbs)):
    x = gatherCollocationData(lbs[i], ubs[i], 200)
    points.append(x)

interval = 0
x = points[interval]
iterRemaining = 50
iter_num_sum = 0
start_time = time.time()
import warnings
warnings.filterwarnings("ignore")

switchAt = []


iter_num_sum = 0
start_time = time.time()
for i in range(max_iter):
    GACGD_optimizer.zero_grad()

    loss_pde, pde_diff = PINN_GACGD.loss_PDE(x)
    loss_ic, ic_diff = PINN_GACGD.loss_IC()
    loss_bc, loss_bc_u, loss_bc_u_x, bc_diff, bc_diff_x = PINN_GACGD.loss_BC()

    # D_output_bc = discriminator(, PINN_GACGD.t_bc)
    D_output_ic = discriminator(PINN_GACGD.x_ic_tensor, PINN_GACGD.t_ic_tensor)[:, [0]]
    D_output_pde = discriminator(torch.from_numpy(x[:, 0, None]).to(device), torch.from_numpy(x[:, 1, None]).to(device))[:, [1]]
    D_output_bc_1 = discriminator(PINN_GACGD.ones_tensor, PINN_GACGD.t_bc)[:, [2]]
    D_output_bc_2 = discriminator(PINN_GACGD.ones_tensor, PINN_GACGD.t_bc)[:, [3]]

    CPINNloss_ic = D_output_ic * ic_diff
    CPINNloss_pde = D_output_pde * pde_diff
    CPINNloss_bc = D_output_bc_1 * bc_diff
    CPINNloss_bc_x = D_output_bc_2 * bc_diff_x

    CPINNloss = CPINNloss_bc.mean() + CPINNloss_bc_x.mean() + CPINNloss_pde.mean() + CPINNloss_ic.mean()
    CPINNloss_max = -CPINNloss
    GACGD_optimizer.step(CPINNloss, CPINNloss_max)

    iter_num_sum += GACGD_optimizer.info["num_iter"]


    if (loss_pde.item() < adaptiveTol and interval != 9):
      
        interval += 1
        switchAt.append([interval, i])
        np.savetxt(f"{path}/history/Interval_Info.csv", switchAt)
        # print(f"Enter interval {interval} at {i} iteration, loss_pde = {loss_pde.item()}")
        x = np.vstack((x, points[interval]))


    if i % recordPer == 0:
        sample_loss, sample_loss_pde, loss_ic, loss_bc = PINN_GACGD.loss(x)

        loss, loss_pde, loss_ic, loss_bc = PINN_GACGD.loss()

        error_vec, u_pred = PINN_GACGD.test(graphPer != 0 and i % graphPer == 0)
        new_row = {"iter":i, 
              "L2 Error": error_vec, 
              "PINN loss": loss.item(), 
              "subset PINN loss": sample_loss.item(), 
              "subset PINN pde loss": sample_loss_pde.item(), 
              "PINN BC loss": loss_bc.item(), 
              "PINN PDE loss": loss_pde.item(), 
              "CPINN loss": CPINNloss.item(), 
              "CPINN BC loss": CPINNloss_bc.mean().item(), 
              "CPINN BC loss_x": CPINNloss_bc_x.mean().item(), 
              "CPINN PDE loss": CPINNloss_pde.mean().item(), 
              "iter_num_sum": iter_num_sum
              }
        ACGDInfo = ACGDInfo.append(new_row, ignore_index = True)
        print(f"iter: {i}, L2 relative error: {error_vec}, PINN loss: {loss}, loss_pde: {loss_pde.item()}, loss_bc: {loss_bc.item()}, loss_ic: {loss_ic.item()}")
    if i % savePer == 0:
        
        ACGDInfo.to_csv(f"{path}/history/ACGDInfo_iter_{i}.csv")
        np.savetxt(f"{path}/prediction/u_pred_iter_{i}.csv", u_pred)

        torch.save({
            "PINN_state_dict": PINN_GACGD.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_state_dict": GACGD_optimizer.state_dict()
            }, path + f"/models/PINNACGD_model_iter_{i}.pt")
        
    

print("Training Time: ", time.time() - start_time)