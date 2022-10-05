import numpy as np

import scipy.io
from pyDOE import lhs
# from plotting import newfig, savefig
import time
import torch
import torch.nn as nn
import torch.autograd as autograd
from Schrodinger_networks import PINN_Schrodinger
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
print(device)
torch.set_default_tensor_type(torch.FloatTensor)



import argparse
parser = argparse.ArgumentParser(description='Enter the parameters')
parser.add_argument('-lr','--lr', help='learning rate', type = float, required=True)
parser.add_argument('-pinn','--pinn', help='PINN layers and neurons', nargs="+", type=int, required=True)

args = parser.parse_args()


# PINN set up
PINNlayers = [2, args.pinn[0], args.pinn[1], 2]

# customize the training
lr = args.lr
PINN = PINN_Schrodinger(x0, u0, v0, tb, X_f, PINNlayers, lb, ub).to(device).float()


path = f"Schrodinger/output/ToAdd_Adam_lr_{lr}_PINN_{PINNlayers}"

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
        

adam = torch.optim.Adam(PINN.parameters(), lr = lr)

max_iter = 6200001
AdamInfo = []
print(PINN)
print(adam)


start_time = time.time()

for i in range(max_iter):
    adam.zero_grad()
    loss, loss0, lossb, lossf = PINN.loss()
    loss.backward()
    adam.step()

    if i % 100 == 0:
        error_u, error_v, error_h, u_pred, v_pred = PINN.test(X_star, u_star, v_star)
        AdamInfo.append([i, error_u.item(), error_v.item(), error_h.item(), loss.item(), loss0.item(), lossb.item(), lossf.item()])
        # print(f"iter {i}, loss: {loss.item()}, error_u: {error_u.item()}, error_v: {error_v.item()}， error_h: {error_h.item()}")

    if i % 2000000 == 0 or i == max_iter - 1:
            np.savetxt(path + f"/history/iter_{i}.csv", AdamInfo)
            np.savetxt(path + f"/prediction/u_pred_iter_{i}.csv", u_pred)
            np.savetxt(path + f"/prediction/v_pred_iter_{i}.csv", v_pred)
            torch.save({
                "PINN_state_dict": PINN.state_dict(),
                "optimizer_state_dict": adam.state_dict()
                }, path + f"/models/state_dict_iter_{i}.pt")
            
            