import numpy as np
import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import trainingData, testingData
import extragradient
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
PINNExtraSGD = networks.PINN_Poisson_2d(layers, x_test, y_test, u_test, f_test,
                                xy_bc[:,[0]], xy_bc[:,[1]], u_bc, 
                                f_xy, xy_inside[:,[0]], xy_inside[:,[1]], 1234)


PINNExtraSGD.to(device)
print(PINNExtraSGD)

DExtraSGD = networks.Discriminator(2, 25 ,2)
DExtraSGD.to(device)
max_iter = 8500001
lr = 0.01

dis_optimizer = extragradient.ExtraSGD(DExtraSGD.parameters(), lr=lr)
PINN_optimizer = extragradient.ExtraSGD(PINNExtraSGD.parameters(), lr=lr)

n_iteration_t = 0 #keeps track of whether perform extrapolation or call step()
recordPer = 10000
savePer = 500000
graphPer = 0
# extraSGDInfo = np.empty(((int)(max_iter / recordPer) + 1, 5)) #iterCount, L2 error, PINN loss, loss_pde, loss_bc
extraSGDInfo = []

start_time = time.time()
for i in range(max_iter):

        
        
        loss1 = DExtraSGD(PINNExtraSGD.x_inside_train, PINNExtraSGD.y_inside_train)[:,[0]] * PINNExtraSGD.test_PDE()
        loss2 = DExtraSGD(PINNExtraSGD.x_bc, PINNExtraSGD.y_bc)[:,[1]] * (PINNExtraSGD(PINNExtraSGD.x_bc, PINNExtraSGD.y_bc) - PINNExtraSGD.u_bc)
        loss = loss1.mean() + loss2.mean()
        dis_loss = -loss.clone()
        
        
        if i % 2 ==0:
            for p in PINNExtraSGD.parameters():
                p.requires_grad = False
            dis_optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
                
            dis_loss.backward(retain_graph = True)
            
            dis_optimizer.extrapolation()

            for p in PINNExtraSGD.parameters():
                p.requires_grad = True
            for p in DExtraSGD.parameters():
                p.requires_grad = False
            
            PINN_optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
                
            loss.backward(retain_graph = True)
            
            PINN_optimizer.extrapolation()
            
            for p in DExtraSGD.parameters():
                p.requires_grad = True
        else:
            for p in PINNExtraSGD.parameters():
                p.requires_grad = False
                
            dis_optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
                
            dis_loss.backward(retain_graph = True)
            
            dis_optimizer.step()

            for p in PINNExtraSGD.parameters():
                p.requires_grad = True
            for p in DExtraSGD.parameters():
                p.requires_grad = False
                
            PINN_optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
            
            loss.backward(retain_graph = True)
            
            PINN_optimizer.step()
            
            for p in DExtraSGD.parameters():
                p.requires_grad = True
        
        if i % recordPer == 0:
            g_loss, loss_bc, loss_pde = PINNExtraSGD.loss()
            error_vec = 0
            if graphPer != 0 and i % graphPer == 0:
                error_vec, _ = PINNExtraSGD.test(True)
            else:
                error_vec, _ = PINNExtraSGD.test(False)
                
            extraSGDInfo.append([i, error_vec, loss.item(), g_loss.item(), loss_pde.item()])

            # print(f"iter: {i}, PINN loss: {g_loss}, L2 relative error: {error_vec}, loss_pde: {loss_pde.item()}, loss_bc: {loss_bc.item()} "
            #      + f"composite loss: {loss.item()}, loss1: {loss1.mean().item()}, loss2: {loss2.mean().item()}")
        if i % savePer == 0:
            csv_path = f"Poisson/output/NewExtraSGD/ExtraSGDInfo_iter_{i}.csv"
            np.savetxt(csv_path, extraSGDInfo)