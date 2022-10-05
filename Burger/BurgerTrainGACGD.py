import numpy as np
import torch
import time
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import scipy.io
from pyDOE import lhs
data = scipy.io.loadmat('Burger/burgers_shock.mat')


N_u = 100
N_f = 10000

np.random.seed(123)

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x,t)

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.flatten()[:,None]              

# Doman bounds
lb = X_star.min(0)
ub = X_star.max(0)    
    
xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T)) #[x, 0], 256 * 2
uu1 = Exact[0:1,:].T #256 * 1
xx2 = np.hstack((X[:,0:1], T[:,0:1])) #[-1, t], 100 * 2
uu2 = Exact[:,0:1] #0 100 * 1
xx3 = np.hstack((X[:,-1:], T[:,-1:])) #[1, t], 100 * 2
uu3 = Exact[:,-1:] #0 100 * 1

X_u_train = np.vstack([xx1, xx2, xx3])
X_f_train = lb + (ub-lb)*lhs(2, N_f) # for PDE constraint
X_f_train = np.vstack((X_f_train, X_u_train))
u_train = np.vstack([uu1, uu2, uu3])

idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
X_u_train = X_u_train[idx, :]
u_train = u_train[idx,:]

nu = 0.01/np.pi

import torch.nn as nn
import torch.autograd as autograd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PINN_Burger(nn.Module):
    def __init__(self, X_u, u_train, X_f, layers, lb, ub, nu, X_test, u_test):
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
        
        self.lb = torch.from_numpy(lb).to(device)
        self.ub = torch.from_numpy(ub).to(device)
    
        self.x_u = torch.from_numpy(X_u[:,0:1]).to(device) # bc
        self.t_u = torch.from_numpy(X_u[:,1:2]).to(device) # bc
        
        self.x_f = torch.from_numpy(X_f[:,0:1]).to(device) # pde
        self.t_f = torch.from_numpy(X_f[:,1:2]).to(device) # pde
        
        self.u_train = torch.from_numpy(u_train).to(device)
        
        self.x_test = torch.from_numpy(X_test).to(device)
        self.u_test = torch.from_numpy(u_test).to(device)
      
        self.layers = layers
        self.nu = nu
        
        # print(self.x_u.shape)
        # print(self.t_u.shape)
        # print(self.x_f.shape)
        # print(self.t_u.shape)
        # print("u_test: ", self.u_test.shape)
        'foward pass'
    def forward(self, x, t):
        
        if torch.is_tensor(x) != True:
          print("convert x to tensor")
          _x = torch.from_numpy(x).to(device)
        else:
          _x = x.clone()
        if torch.is_tensor(t) != True:         
          _t = torch.from_numpy(t).to(device)
        else:
          _t = t.clone()
        # print("forward() x shape", x.shape)
        # print("forward() t shape", t.shape)


        a = torch.cat((_x, _t), 1)
        
        a = 2.0 * (a - self.lb)/(self.ub - self.lb) - 1.0

        # print(a.shape)

        for i in range(len(self.linears)-1):
            
            z = self.linears[i](a)
                        
            a = self.activation(z)
            
        a = self.linears[-1](a)
        
        return a

    def loss_BC(self):
        '''
        the parameters must be all None or all non-None 
        '''
        _u_output = self.forward(self.x_u, self.t_u)
        loss_bc = self.loss_function(_u_output, self.u_train)
        return loss_bc, self.u_train - _u_output
    
    # We do not want to hard code the right hand side. Instead, we want to pass it as a label, just like the boundary conditions
    def loss_PDE(self):
        '''
        the parameters must be all None or all non-None 
        '''
        shouldBeZero = self.PDE_residual()
        
        loss_f = self.loss_function(shouldBeZero, torch.zeros_like(shouldBeZero))
                
        return loss_f

    def PDE_residual(self):
        '''
        the parameters must be all None or all non-None 
        '''
        
        _x = self.x_f.clone()
        _t = self.t_f.clone()

        _x.requires_grad = True
        _t.requires_grad = True
        
        u = self.forward(_x, _t)

        u_x = autograd.grad(u,_x,torch.ones([_x.shape[0], 1]).to(device), retain_graph= True, create_graph=True)[0]
        u_t = autograd.grad(u,_t,torch.ones([_t.shape[0], 1]).to(device), retain_graph= True, create_graph=True)[0]

        u_xx = autograd.grad(u_x,_x,torch.ones((_x.shape[0], 1)).to(device), retain_graph= True, create_graph = True)[0]

        shouldBeZero = u_t + u * u_x + - self.nu * u_xx
        shouldBeZero.retain_graph = True
        
        return shouldBeZero

    def loss(self):
        '''
            if no resampling is needed in training iterations, do not provide values for the parameters
        
            the first 3 parameters must be all None or all non-None 
            
            the last 3 parameters must be all None or all non-None 
        
        '''
        
        loss_u, bcError = self.loss_BC()
            
        loss_f = self.loss_PDE()
        
        loss_val = loss_u + loss_f

        return loss_val, loss_u, loss_f
         
    # I am not sure this function should be part of the neural network, and if it is, it should again work with a general f
    # 'test neural network'
    def test(self, graph = False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        _x = self.x_test[:, 0][:, None].clone()
        _t = self.x_test[:, 1][:, None].clone()
        
        
        _x.requires_grad = True
        _t.requires_grad = True
        
        u_pred = self.forward(_x, _t)
        
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
        error_vec = np.linalg.norm((self.u_test.cpu().detach().numpy() - u_pred), 2) / np.linalg.norm(self.u_test.cpu().detach().numpy(), 2)  # Relative L2 Norm of the error (Vector)
        
        u_pred = np.reshape(u_pred, (100, 256), order='C')
        
#         if (graph):
# #           fig, ax = plt.subplots(1, 4,figsize=(30,4))
          
# # prediction plot
#           fig, ax = plt.subplots(1, 3, figsize=(22,4))

#           ax[0].set_title("u_pred")
#           h = ax[0].imshow(u_pred.T, interpolation='nearest', cmap='rainbow', 
#                         extent=[t.min(), t.max(), x.min(), x.max()], 

#                         origin='lower', aspect='auto')
#           divider = make_axes_locatable(ax[0])
#           cax = divider.append_axes("right", size="5%", pad=0.05)
#           fig.colorbar(h, cax=cax)

# # true difference plot
#           ax[1].set_title("u_true")
#           h = ax[1].imshow(Exact.T, interpolation='nearest', cmap='rainbow', 
#                         extent=[t.min(), t.max(), x.min(), x.max()], 
#                         origin='lower', aspect='auto')
#           divider = make_axes_locatable(ax[1])
#           cax = divider.append_axes("right", size="5%", pad=0.05)
#           fig.colorbar(h, cax=cax)

#           ax[2].set_title("u_pred - u_true")
#           # h = ax[2].imshow(np.log(np.abs(u_pred.T - Exact.T)), interpolation='nearest', cmap='rainbow', 
#           h = ax[2].imshow(u_pred.T - Exact.T, interpolation='nearest', cmap='rainbow',
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

import argparse
parser = argparse.ArgumentParser(description='Enter the parameters')
parser.add_argument('-lrmin', help='PINN lr', type = float, required=True)
parser.add_argument('-lrmax', help='discriminator lr', type = float, required=True)
parser.add_argument('-disLayer', help='discriminator layer', type = int, required=True)
parser.add_argument('-disNeu', help='discriminator neuron countp er layer', type = int, required=True)
parser.add_argument('-pinnLayer', help='PINN layer', type = int, required=True)
parser.add_argument('-pinnNeu', help='PINN neuron countp er layer', type = int, required=True)

args = parser.parse_args()

dis_layers = [2, args.disLayer, args.disNeu, 2]
discriminator = Discriminator(dis_layers).to(device).double()
print(discriminator)


PINNlayers = [2, args.pinnLayer, args.pinnNeu, 1]
PINN_GACGD = PINN_Burger(X_u_train, u_train, X_f_train, PINNlayers, lb, ub, nu, X_star, u_star).to(device).double()
print(PINN_GACGD)

import CGDs

lr_max = args.lrmax
lr_min = args.lrmin

tol = 1e-7
atol = 1e-20

GACGD_optimizer = CGDs.GACGD(x_params = discriminator.parameters(), y_params = PINN_GACGD.parameters(),
            lr_x=lr_max, lr_y=lr_min, tol=tol, atol = atol, eps=1e-8, beta=0.99, track_cond = lambda x, y: True)

# print(GACGD_optimizer.get_info())

iter_num_sum = 0

recordPer = 100
path = f"Burger/output/Burger_1Dis_GACGD_tol_{tol}_atol_{atol}_lrmax_{lr_max}_lrmin_{lr_min}_PINN_{args.pinnLayer}_{args.pinnNeu}_Dis_{args.disLayer}_{args.disNeu}"

isDirectory = os.path.isdir(path)

if not os.path.isdir(path):
    os.makedirs(path)
    
if not os.path.isdir(path + "/history"):
    os.makedirs(path + "/history")
    
if not os.path.isdir(path + "/prediction"):
    os.makedirs(path + "/prediction")

if not os.path.isdir(path + "/models"):
    os.makedirs(path + "/models")

cols = ["iter", "L2 Error", "PINN loss", "PINN BC loss", "PINN PDE loss", "CPINN loss", "CPINN BC loss", "CPINN PDE loss", "iter_num", "iter_num_sum"]
GACGDInfo = pd.DataFrame(columns = cols)
print(GACGDInfo.head())
max_iter = 28001
savePer = 500
for i in range(max_iter):
    GACGD_optimizer.zero_grad()

    PDE_residual = PINN_GACGD.PDE_residual()
    loss_bc, bc_diff = PINN_GACGD.loss_BC()

    D_output_bc = discriminator(PINN_GACGD.x_u, PINN_GACGD.t_u)[:,[0]]
    D_output_pde = discriminator(PINN_GACGD.x_f, PINN_GACGD.t_f)[:,[1]]

    CPINNloss_bc = D_output_bc * bc_diff
    CPINNloss_pde = D_output_pde * PDE_residual

    CPINNloss = CPINNloss_bc.mean() + CPINNloss_pde.mean()
    CPINNloss_max = -CPINNloss
    GACGD_optimizer.step(CPINNloss, CPINNloss_max)

    iter_num = GACGD_optimizer.info["num_iter"]
    iter_num_sum += iter_num

    if i % recordPer == 0:
      loss, loss_bc, loss_pde = PINN_GACGD.loss()
      error_vec, u_pred = PINN_GACGD.test(i % 200 == 0)
      new_row = {"iter":i, 
              "L2 Error": error_vec, 
              "PINN loss": loss.item(), 
              "PINN BC loss": loss_bc.item(), 
              "PINN PDE loss": loss_pde.item(), 
              "CPINN loss": CPINNloss.item(), 
              "CPINN BC loss": CPINNloss_bc.mean().item(), 
              "CPINN PDE loss": CPINNloss_pde.mean().item(), 
              "iter_num": iter_num,
              "iter_num_sum": iter_num_sum
              }
      GACGDInfo = GACGDInfo.append(new_row, ignore_index = True)


    if i % savePer == 0:
        GACGDInfo.to_csv(f"{path}/history/GACGDInfo_iter_{i}.csv")
        np.savetxt(f"{path}/prediction/u_pred_iter_{i}.csv", u_pred)
        torch.save({
                "PINN state": PINN_GACGD.state_dict(),
                "discriminator state": discriminator.state_dict(), 
                "optimizer state": GACGD_optimizer.state_dict()
                }
                , f"{path}/models/model_states_iter_{i}.pt")
      # GACGDInfo.append([i, error_vec, loss.item(), loss1.mean().item(), loss2.mean().item(), PINNloss.item(), loss_pde.item(), loss_bc.item(), iter_num_sum])
    #   print(f"iter: {i}, L2 relative error: {error_vec}, PINN loss: {loss.item()}, CPINN loss: {CPINNloss.item()} CPINN_loss_pde: {CPINNloss_pde.mean().item()}, CPINN_loss_bc: {CPINNloss_bc.mean().item()}, iter_num_sum: {iter_num_sum}.")
    