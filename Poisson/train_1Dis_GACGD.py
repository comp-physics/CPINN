import numpy as np
import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from utils import trainingData, testingData
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.DoubleTensor)
print(device)

if device == 'cuda': 
    print(torch.cuda.get_device_name()) 
    
lb = np.array([-2, -2])
ub = np.array([2, 2])

num_bc = 50
num_f= 5000

# import argparse
# parser = argparse.ArgumentParser(description='Enter the parameters')
# parser.add_argument('-tol','--tol', help='relative tolerance for GMRES', type = float, required=True)
# parser.add_argument('-atol','--atol', help='absolute tolerance for GMRES', type = float, required=True)
# parser.add_argument('-g_iter','--g_iter', help='maximum iteration within GMRES', type = int, required=True)

# args = parser.parse_args()

# tol = args.tol
# atol = args.atol
# g_iter = args.g_iter

tol = 1e-7
atol = 1e-20
g_iter = 1000

path = f"Poisson/output/1_dis_GACGD_tol_{tol}_atol_{atol}_g_iter_{g_iter}"

isDirectory = os.path.isdir(path)

if not os.path.isdir(path):
    os.makedirs(path)
    
if not os.path.isdir(path + "/history"):
    os.makedirs(path + "/history")
    
if not os.path.isdir(path + "/models"):
    os.makedirs(path + "/models")
    
if not os.path.isdir(path + "/prediction"):
    os.makedirs(path + "/prediction")
        
        
        
u = lambda xy: np.sin(xy[:, 0]) * np.cos([xy[:, 1]]) # this cannot be a torch function, otherwise the gradient would be recorded
f = lambda x, y: -2 * np.sin(x) * np.cos(y) #torch->np

RNG_key = 123

all_xy_train, xy_bc, u_bc, xy_inside, f_xy = trainingData(lb, ub, num_bc, num_f, u, f, RNG_key)

u_test_method = lambda x, y: np.sin(x) * np.cos(y) #takes 2 inputs, but should return same values as previous u
x_test, y_test, xy_test, u_test, f_test, X, Y, U = testingData(lb, ub, u_test_method, f, 256)



import networks
layers = np.array([2,50 ,50, 50,1])


'ACGD Optimizer'
import CGDs
from training import trainACGD

PINNGACGD = networks.PINN_Poisson_2d(layers, x_test, y_test, u_test, f_test,
                                xy_bc[:,[0]], xy_bc[:,[1]], u_bc, 
                                f_xy, xy_inside[:,[0]], xy_inside[:,[1]], RNG_key)

PINNGACGD.to(device)


D_GACGD = networks.Discriminator(2, 25 ,2)
D_GACGD.to(device)

print(PINNGACGD)
print(D_GACGD)

lr = 0.001

track_cond = lambda x, y:  True
optimizer = CGDs.GACGD(x_params=D_GACGD.parameters(), y_params = PINNGACGD.parameters(), max_iter = g_iter,
            lr_x=lr, lr_y=lr, tol=tol, atol = atol, eps=1e-8, beta=0.99, track_cond = track_cond)

max_iter = 34001
recordPer = 100
savePer = 2000
graphPer = 0

GACGDInfo = []
iter_num_sum = 0
start_time = time.time()

ACGDInfo = pd.DataFrame()

for e in range(max_iter):
    optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
    D_output = D_GACGD(PINNGACGD.x_inside_train, PINNGACGD.y_inside_train) #output[0]=bc, output[1]=inside
    g_pde_diff = PINNGACGD.test_PDE()

    # Optimizer step
    loss1 = D_output[:,[0]] * g_pde_diff
    loss2 = D_GACGD(PINNGACGD.x_bc, PINNGACGD.y_bc)[:,[1]] * (PINNGACGD(PINNGACGD.x_bc, PINNGACGD.y_bc) - PINNGACGD.u_bc)

    loss_y = loss1.mean() + loss2.mean()
    
    loss_x = -loss_y
    optimizer.step(loss_x, loss_y, 0) #breaks at first step() call
    # print(e) 




    iter_num_sum += optimizer.info["num_iter"]
    if e % recordPer == 0:
    #   losses.append(loss.item())
      g_loss, loss_bc, g_pde_loss = PINNGACGD.loss()
      error_vec, u_pred = PINNGACGD.test(graphPer != 0 and e % graphPer == 0)
      
      ACGDInfo = ACGDInfo.append({
        "iter": e,
        "L2 error": error_vec,
        "PINN loss": g_loss.item(),
        "PINN BC loss": loss_bc.item(),
        "PINN PDE loss": g_pde_loss.item(),
        "CPINN loss": loss_y.item(),
        "CPINN PDE loss": loss1.mean().item(),
        "CPINN BC loss": loss2.mean().item(),
        "iter_num_sum" : iter_num_sum
        }, ignore_index = True)

    if e % savePer == 0: 
        ACGDInfo.to_csv(f"{path}/history/ACGDInfo_{e}.csv")
        np.savetxt(f"{path}/prediction/PINNPrediction_iter_{e}.csv", u_pred)
        torch.save({
            "PINN_state_dict": PINNGACGD.state_dict(),
            "Discriminator_state_dict": D_GACGD.state_dict(),
            "GACGD_optimizer_state_dict" : optimizer.state_dict(),
        }, f"{path}/models/GACGD_models_iter_{e}.pt")

print(f"Time: {time.time() - start_time}")
