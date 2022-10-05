import numpy as np
import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import trainingData, testingData
from training import trainNonCGD

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.DoubleTensor)
print(device)

if device == 'cuda': 
    print(torch.cuda.get_device_name()) 
    
lb = np.array([-2, -2])
ub = np.array([2, 2])

num_bc = 50
num_f= 5000

u = lambda xy: np.sin(xy[:, 0]) * np.cos([xy[:, 1]]) # this cannot be a torch function, otherwise the gradient would be recorded
f = lambda x, y: -2 * np.sin(x) * np.cos(y) #torch->np
all_xy_train, xy_bc, u_bc, xy_inside, f_xy = trainingData(lb, ub, num_bc, num_f, u, f, 123)

u_test_method = lambda x, y: np.sin(x) * np.cos(y) #takes 2 inputs, but should return same values as previous u
x_test, y_test, xy_test, u_test, f_test, X, Y, U = testingData(lb, ub, u_test_method, f, 256)




import networks
layers = np.array([2,50 ,50, 50,1])
# printMemory()
#(self, layers, x_test, y_test, u_test, x_bc, y_bc, u_bc, fxy, x_inside_train, y_inside_train):

path = f"Poisson/output/NewSGD"


isDirectory = os.path.isdir(path)

if not os.path.isdir(path):
    os.makedirs(path)
    
if not os.path.isdir(path + "/history"):
    os.makedirs(path + "/history")
    
if not os.path.isdir(path + "/prediction"):
    os.makedirs(path + "/prediction")
    

PINN = networks.PINN_Poisson_2d(layers, x_test, y_test, u_test, f_test,
                                xy_bc[:,[0]], xy_bc[:,[1]], u_bc, 
                                f_xy, xy_inside[:,[0]], xy_inside[:,[1]])


PINN.to(device)
print(PINN)


'SGD Optimizer'
lr = 0.01
optimizer = optim.SGD(PINN.parameters(), lr=lr)
recordPer = 20000
max_iter = 21500001
_, SGDInfo, SGDSamples = trainNonCGD(PINN, optimizer, max_iter = max_iter, recordPer = recordPer, graphPer = 0, path = path, savePer = 500000,
                          miniBatch = False, batchSizeBC = num_bc, batchSizePDE = num_f, lb = lb, ub = ub, u = u, f = f)



# np.savetxt(csv_path, SGDInfo)