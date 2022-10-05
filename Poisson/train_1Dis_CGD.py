import numpy as np
import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import trainingData, testingData
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
PINNBCGD = networks.PINN_Poisson_2d(layers, x_test, y_test, u_test, f_test,
                                xy_bc[:,[0]], xy_bc[:,[1]], u_bc, 
                                f_xy, xy_inside[:,[0]], xy_inside[:,[1]])

# num_bc_4 = 200
# num_f_4= 20000
# all_xy_train_4, xy_bc_4, u_bc_4, xy_inside_4, f_xy_4 = trainingData(lb, ub, num_bc_4, num_f_4, u, f)

# PINN_4 = networks.PINN_Poisson_2d(layers, x_test, y_test, u_test, f_test,
#                                 xy_bc_4[:,[0]], xy_bc_4[:,[1]], u_bc_4, 
#                                 f_xy_4, xy_inside_4[:,[0]], xy_inside_4[:,[1]])

# num_bc_10 = 500
# num_f_10= 50000
# all_xy_train_10, xy_bc_10, u_bc_10, xy_inside_10, f_xy_10 = trainingData(lb, ub, num_bc_10, num_f_10, u, f)

# PINN_10 = networks.PINN_Poisson_2d(layers, x_test, y_test, u_test, f_test,
#                                 xy_bc_4[:,[0]], xy_bc_4[:,[1]], u_bc_4,
#                                 f_xy_4, xy_inside_4[:,[0]], xy_inside_4[:,[1]])
PINNBCGD.to(device)
print(PINNBCGD)

D = networks.Discriminator(2, 25 ,2)
D.to(device)
print(D)
# printMemory()



'CGD Optimizer'
import CGDs
import importlib
importlib.reload(CGDs)
# BCGDInfo = np.empty((IterErrorCount, 5)) #2 if not using CGD, 5 if using CGD

max_iter = 600001
graphPer = 0
iter_recordBCGD = 200

savePer = 5000
# Info = np.empty(((int)(max_iter / recordPer), 8)) #[i, error_vec, loss.item(), g_loss.item(), loss_pde.mean().item(), iter_num, iter_num_sum]

# BCGDInfo = np.empty(((int)(max_iter / iter_recordBCGD) + 1, 7)) #[i, error_vec, loss.item(), g_loss.item(), loss_pde.mean().item(), iter_num, iter_num_sum]
BCGDInfo = []


D_BCGD = networks.Discriminator(2, 25 ,2)
D_BCGD.to(device)
D_BCGD.load_state_dict(D.state_dict()) # copy weights and stuff

print(PINNBCGD)
print(D_BCGD)

optimizer = CGDs.BCGD(max_params=D_BCGD.parameters(), min_params= PINNBCGD.parameters(), device = device,
                 lr_max=0.02, lr_min=0.02, tol=1e-10, collect_info=True)
lossFunction = torch.nn.MSELoss()

start_time = time.time()
iter_num_sum = 0
for e in range(max_iter):
    optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
    # g_diff = PINNBCGD.test_PDE(inside_x, inside_y)

    D_output = D_BCGD(PINNBCGD.x_inside_train, PINNBCGD.y_inside_train) #output[0]=bc, output[1]=inside

    g_pde_diff = PINNBCGD.test_PDE()

    # Optimizer step
    loss1 = D_output[:,[0]] * g_pde_diff
    loss2 = D_BCGD(PINNBCGD.x_bc, PINNBCGD.y_bc)[:,[1]] * (PINNBCGD(PINNBCGD.x_bc, PINNBCGD.y_bc) - PINNBCGD.u_bc)

    loss = loss1.mean() + loss2.mean()
    # loss = loss1.mean()+loss2.mean()
    optimizer.step(loss)
    # print(e)
    iter_num_sum += optimizer.get_info()["iter_num"]
    if e % iter_recordBCGD == 0:
      # losses.append(loss.item())
      g_loss, loss_bc, g_pde_loss = PINNBCGD.loss()
      error_vec, _ = PINNBCGD.test(graphPer != 0 and e % graphPer == 0)
      iter_num  = optimizer.get_info()["iter_num"]
      # print('Epoch :{}, PINN error: {}, Total Loss: {}, loss_real: {}, loss_fake: {}'.format(e, error_vec, loss.item(), loss_real.item(), loss_fake.item()))
    #   print('Epoch :{}, PINN loss: {}, PINN error: {}, loss1: {}:, loss2: {}, total loss: {}, iter_num: {}, cumulative iter_num: {}'.format(e, g_loss.item(), error_vec, loss1.mean().item(), loss2.mean().item(), loss.item(), iter_num, iter_num_sum))
     #[i, error_vec, loss.item(), g_loss.item(), loss_pde.mean().item(), iter_num, iter_num_sum]
      BCGDInfo.append([e, error_vec, loss.item(), g_loss.item(), g_pde_loss.mean().item(), iter_num, iter_num_sum])
    if e % savePer == 0:
        np.savetxt(f"Poisson/output/CGD/CGDInfo_iter_{e}.csv", BCGDInfo)