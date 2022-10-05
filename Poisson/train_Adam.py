import numpy as np
import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import trainingData, testingData, PINNplot, ACGDPlot
from training import trainNonCGD
import os 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.DoubleTensor)
print(device)

import argparse
parser = argparse.ArgumentParser(description='Enter the parameters')
parser.add_argument('-rng','--rng', help='rng key', type = int, required=False)
args = parser.parse_args()


RNG_key = args.rng
path = f"Poisson/output/PINN_Adam_RNG_{RNG_key}"

isDirectory = os.path.isdir(path)

if not os.path.isdir(path):
    os.makedirs(path)
    
if not os.path.isdir(path + "/history"):
    os.makedirs(path + "/history")
    
if not os.path.isdir(path + "/models"):
    os.makedirs(path + "/models")
    
if not os.path.isdir(path + "/prediction"):
    os.makedirs(path + "/prediction")
    
if device == 'cuda': 
    print(torch.cuda.get_device_name()) 
    
lb = np.array([-2, -2])
ub = np.array([2, 2])

num_bc = 50
num_f= 5000

u = lambda xy: np.sin(xy[:, 0]) * np.cos([xy[:, 1]]) # this cannot be a torch function, otherwise the gradient would be recorded
f = lambda x, y: -2 * np.sin(x) * np.cos(y) #torch->np
all_xy_train, xy_bc, u_bc, xy_inside, f_xy = trainingData(lb, ub, num_bc, num_f, u, f, RNG_key)

u_test_method = lambda x, y: np.sin(x) * np.cos(y) #takes 2 inputs, but should return same values as previous u
x_test, y_test, xy_test, u_test, f_test, X, Y, U = testingData(lb, ub, u_test_method, f, 256)




import networks
layers = np.array([2,50 ,50, 50,1])
# printMemory()
#(self, layers, x_test, y_test, u_test, x_bc, y_bc, u_bc, fxy, x_inside_train, y_inside_train):
PINN = networks.PINN_Poisson_2d(layers, x_test, y_test, u_test, f_test,
                                xy_bc[:,[0]], xy_bc[:,[1]], u_bc, 
                                f_xy, xy_inside[:,[0]], xy_inside[:,[1]], RNG_key)


PINN.float().to(device)
print(PINN)


'Adam Optimizer'

PINNAdam = networks.PINN_Poisson_2d(layers, x_test, y_test, u_test, f_test,
                                xy_bc[:,[0]], xy_bc[:,[1]], u_bc, 
                                f_xy, xy_inside[:,[0]], xy_inside[:,[1]], RNG_key)
PINNAdam.to(device)

PINNAdam.load_state_dict(PINN.state_dict()) # copy weights and stuff

betas = (0.99, 0.99)
optimizer = optim.Adam(PINNAdam.parameters(), lr=0.001,betas=betas, eps=1e-08, weight_decay=0, amsgrad=False)
recordPer =  10000

max_iter = 30000001
lr = 0.001
_, AdamInfo, AdamSamples = trainNonCGD(PINNAdam, optimizer, max_iter, recordPer=recordPer, graphPer = 0, path = path,
                      miniBatch = False, batchSizeBC = num_bc, batchSizePDE = num_f, lb = lb, ub = ub, u = u, f = f, trainBatchFor = 0)
                      
error, u_pred = PINNAdam.test(False)


np.savetxt(path + f"history/Adam_Info_{max_iter}.csv", AdamInfo)
np.savetxt(path + f"prediction/Adam_prediction_{max_iter}.csv", u_pred)
torch.save({
    "PINN_state_dict": PINNAdam.state_dict(),
    "Adam_state_dict": optimizer.state_dict()
}, path + f"models/Adam_models_{max_iter}.pt")



