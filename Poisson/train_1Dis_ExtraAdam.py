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
PINNExtraAdam = networks.PINN_Poisson_2d(layers, x_test, y_test, u_test, f_test,
                                xy_bc[:,[0]], xy_bc[:,[1]], u_bc, 
                                f_xy, xy_inside[:,[0]], xy_inside[:,[1]], 1234)


PINNExtraAdam.to(device)
print(PINNExtraAdam)

DExtraAdam = networks.Discriminator(2, 25 ,2)
DExtraAdam.to(device)

max_iter = 8500001
dis_lr = 5e-4
PINN_lr = 5e-5
betas=(0.3, 0.7)

dis_optimizer = extragradient.ExtraAdam(DExtraAdam.parameters(), lr=dis_lr, betas = betas)
PINN_optimizer = extragradient.ExtraAdam(PINNExtraAdam.parameters(), lr=PINN_lr, betas = betas)

n_iteration_t = 0 #keeps track of whether perform extrapolation or call step()
recordPer = 10000
savePer = 500000
graphPer = 0
# extraAdamInfo = np.empty(((int)(max_iter / recordPer) + 1, 5)) #iterCount, L2 error, PINN loss, loss_pde, loss_bc
extraAdamInfo = []

start_time = time.time()
for i in range(max_iter):
    
        all_xy_train, xy_bc, u_bc, xy_inside, f_xy = trainingData(lb, ub, num_bc, num_f, u, f)
        
        x_bc = xy_bc[:,[0]]
        y_bc = xy_bc[:,[1]]
        x_inside = xy_inside[:,[0]]
        y_inside = xy_inside[:,[0]]
        
        loss1 = DExtraAdam(x_inside, y_inside)[:,[0]] * PINNExtraAdam.test_PDE(x_inside, y_inside, f_xy)
        loss2 = DExtraAdam(x_bc, y_bc)[:,[1]] * (PINNExtraAdam(x_bc, y_bc) - u_bc)
        loss = loss1.mean() + loss2.mean()
        dis_loss = -loss.clone()
        
        
        for p in PINNExtraAdam.parameters():
            p.requires_grad = False
        DExtraAdam.zero_grad()     # zeroes the gradient buffers of all parameters

        dis_loss.backward(retain_graph = True)
        
        if i % 2 ==0:
            dis_optimizer.extrapolation()
        else:
            dis_optimizer.step()
                
        for p in PINNExtraAdam.parameters():
            p.requires_grad = True
        for p in DExtraAdam.parameters():
            p.requires_grad = False

        PINNExtraAdam.zero_grad()     # zeroes the gradient buffers of all parameters

        loss.backward()
        
        if i % 2 ==0:
            PINN_optimizer.extrapolation()
        else:
            PINN_optimizer.step()
        
        for p in DExtraAdam.parameters():
            p.requires_grad = True
            
#         if i % 2 ==0:
#             dis_optimizer.extrapolation()
#         else:
#             dis_optimizer.step()
   
        if i % recordPer == 0:
            g_loss, loss_bc, loss_pde = PINNExtraAdam.loss()
            error_vec = 0
            if graphPer != 0 and i % graphPer == 0:
                error_vec, _ = PINNExtraAdam.test(True)
            else:
                error_vec, _ = PINNExtraAdam.test(False)
                
            extraAdamInfo.append([i, error_vec, loss.item(), g_loss.item(), loss_pde.item()])
        if i % savePer == 0:
            csv_path = f"Poisson/output/NewExtraAdam/ExtraAdamInfo_iter_{i}.csv"
            np.savetxt(csv_path, extraAdamInfo)
