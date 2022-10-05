import numpy as np

import scipy.io
from pyDOE import lhs
# from plotting import newfig, savefig
import time
import torch
import torch.nn as nn
import torch.autograd as autograd
from Schrodinger_networks import PINN_Schrodinger, Discriminator
import os
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK']='True'
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

noise = 0.0        

# Doman bounds
lb = np.array([-5.0, 0.0])
ub = np.array([5.0, np.pi/2])

N0 = 50
N_b = 50
N_f = 20000

data = scipy.io.loadmat('Schrodinger/NLS.mat')

t = data['tt'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = data['uu']
Exact_u = np.real(Exact)
Exact_v = np.imag(Exact)
Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)

X, T = np.meshgrid(x,t)

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact_u.T.flatten()[:,None]
v_star = Exact_v.T.flatten()[:,None]
h_star = Exact_h.T.flatten()[:,None]

###########################
np.random.seed(1234)

idx_x = np.random.choice(x.shape[0], N0, replace=False)
x0 = x[idx_x,:]
u0 = Exact_u[idx_x,0:1]
v0 = Exact_v[idx_x,0:1]

idx_t = np.random.choice(t.shape[0], N_b, replace=False)
tb = t[idx_t,:]

X_f = lb + (ub-lb)*lhs(2, N_f)


tb_tensor = torch.from_numpy(tb).float().to(device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
print(device)
torch.set_default_tensor_type(torch.FloatTensor)



import argparse
parser = argparse.ArgumentParser(description='Enter the parameters')
parser.add_argument('-lrmax','--lrmax', help='discriminator learning rate', type = float, required=True)
parser.add_argument('-lrmin','--lrmin', help='pinn learning rate', type = float, required=True)
parser.add_argument('-pinn','--pinn', help='PINN layers and neurons', nargs="+", type=int, required=True)
parser.add_argument('-dis','--dis', help='Discriminator layers and neurons', nargs="+", type=int, required=True)

args = parser.parse_args()


# PINN set up
PINNlayers = [2, args.pinn[0], args.pinn[1], 2]
Dislayers = [2, args.dis[0], args.dis[1], 8]

# customize the training
lr_max = args.lrmax
lr_min = args.lrmin

PINNGACGD = PINN_Schrodinger(x0, u0, v0, tb, X_f, PINNlayers, lb, ub).to(device).float()
discriminator = Discriminator(Dislayers).to(device).float()

tol = 1e-7
atol = 1e-20
gmres_iter = 500
path = f"Schrodinger/output/Single_Dis_GACGD_tol_{tol}_atol_{atol}_gmresiter_{gmres_iter}_lrmax_{lr_max}_lrmin_{lr_min}_PINN_{PINNlayers}_Dis_{Dislayers}"

isDirectory = os.path.isdir(path)

forceTrain = True # change this if you want to force it to train

if isDirectory and not forceTrain: 
    print("An identical training has been done before. Exit.")
    exit()
else:
    if not os.path.isdir(path):
        os.makedirs(path)
    if not os.path.isdir(path + "/history"):
        os.makedirs(path + "/history")

    if not os.path.isdir(path + "/models"):
        os.makedirs(path + "/models")

    if not os.path.isdir(path + "/prediction"):
        os.makedirs(path + "/prediction")
        



import CGDs

GACGDoptimizer = CGDs.GACGD(x_params = discriminator.parameters(), y_params = PINNGACGD.parameters(),
    lr_x=lr_max, lr_y=lr_min, eps=1e-8, tol = tol, atol = atol, max_iter = gmres_iter,  track_cond = lambda x, y: True)
print(PINNGACGD)
print(discriminator)
print(GACGDoptimizer.state_dict())

GACGDInfo = pd.DataFrame()
max_iter = 4201

D_0_input = torch.cat((PINNGACGD.x0, PINNGACGD.t0), 1).to(device).float()
D_f_input = torch.cat((PINNGACGD.x_f, PINNGACGD.t_f), 1).to(device).float()
D_bc_input = torch.cat((torch.zeros_like(tb_tensor), tb_tensor), 1).to(device).float()

iter_num_sum = 0

start_time = time.time()

for i in range(max_iter):
  GACGDoptimizer.zero_grad()

  PINNoutput0 = PINNGACGD.forward(PINNGACGD.x0, PINNGACGD.t0)
  Doutput0 = discriminator(D_0_input)

  loss_0 = Doutput0[:,[0]] * (PINNoutput0[:, [0]] - PINNGACGD.u0) + Doutput0[:,[1]] * (PINNoutput0[:, [1]] - PINNGACGD.v0)


  u_lb_pred, u_ub_pred, v_lb_pred, v_ub_pred, u_x_lb_pred, u_x_ub_pred, v_x_lb_pred, v_x_ub_pred = PINNGACGD.residual_bc()

  Doutput_bc = discriminator(D_bc_input)
  loss_bc = Doutput_bc[:, [2]] * (u_lb_pred - u_ub_pred) + Doutput_bc[:, [3]] * (v_lb_pred - v_ub_pred)
  
  Doutput_bc_x = discriminator(D_bc_input)
  loss_bc_x = Doutput_bc_x[:, [4]] * (u_x_lb_pred - u_x_ub_pred) + Doutput_bc_x[:, [5]] * (v_x_lb_pred - v_x_ub_pred)


  f_u, f_v = PINNGACGD.residual_f()

  Doutputf = discriminator(D_f_input)
  loss_f = Doutputf[:,[6]] * f_u + Doutputf[:,[7]] * f_v

  loss = loss_0.mean() + loss_bc.mean() + loss_bc_x.mean() + loss_f.mean()
  loss_max = -loss
  GACGDoptimizer.step(loss, loss_max)

  iter_num_sum += GACGDoptimizer.info["num_iter"]
  if i % 50 == 0: 
    error_u, error_v, error_h, u_pred, v_pred = PINNGACGD.test(X_star, u_star, v_star)
    PINNloss, _, _, _ = PINNGACGD.loss()
    # print(f"iter {i}, loss: {loss.item()}, PINN loss: {PINNloss.item()} error_u: {error_u.item()}, error_v: {error_v.item()}， error_h: {error_h.item()}, iter_num_sum: {iter_num_sum}")
    GACGDInfo = GACGDInfo.append({
        "iter": i,
        "u error": error_u.item(),
        "v error": error_v.item(),
        "L2 error": error_h.item(),
        "CPINN loss": loss.item(),
        "PINN loss": PINNloss.item(),
        "iter_num_sum": iter_num_sum
        }, ignore_index = True)
  
  if i % 200 == 0:
            GACGDInfo.to_csv(path + f"/history/iter_{i}.csv")
            # Gnp.savetxt(path + f"/history/iter_{i}.csv", ACGDInfo)
            np.savetxt(path + f"/prediction/u_pred_iter_{i}.csv", u_pred)
            np.savetxt(path + f"/prediction/v_pred_iter_{i}.csv", v_pred)
            torch.save({
                "PINN_state_dict": PINNGACGD.state_dict(),
                "Discriminator_state_dict": discriminator.state_dict(),
                "optimizer_state_dict": GACGDoptimizer.state_dict()
                }, path + f"/models/state_dict_iter_{i}.pt")
            
            