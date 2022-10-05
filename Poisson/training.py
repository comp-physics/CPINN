# This file should contain similar training functions for the remaining algorithms
import numpy as np
import time
from utils import trainingData
import matplotlib.pyplot as plt
import torch

def trainNonCGD(PINN, optimizer, max_iter, recordPer = 500, graphPer = 0, path = "", savePer = 200000,
                miniBatch = False, batchSizeBC = 0, batchSizePDE = 0, lb = None, ub = None, u = None, f = None, trainBatchFor = 1):
    # info = np.empty(((int)(max_iter / recordPer) + 1, 5)) #iterCount, L2 error, PINN loss, loss_pde, loss_bc
    info = []
    start_time = time.time()
    samplePoints = []
    
    _, xy_bc, u_bc, xy_inside, f_xy = trainingData(lb, ub, batchSizeBC, batchSizePDE, u, f)
    for i in range(max_iter):
        loss, loss_bc, loss_pde = 0, 0, 0
        
        if miniBatch:
            if i % trainBatchFor == 0:
                _, xy_bc, u_bc, xy_inside, f_xy = trainingData(lb, ub, batchSizeBC, batchSizePDE, u, f)
            
            samplePoints.append((np.hstack((xy_bc.cpu().detach().numpy()[:, 0], xy_bc.cpu().detach().numpy()[:, 1],)), 
                                np.hstack((xy_inside.cpu().detach().numpy()[:, 0], xy_inside.cpu().detach().numpy()[:, 1]))
                                ))
            loss, loss_bc, loss_pde = PINN.loss(xy_bc[:,[0]], xy_bc[:,[1]], u_bc, xy_inside[:,[0]], xy_inside[:,[1]], f_xy)
        else:
            loss, loss_bc, loss_pde = PINN.loss()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        
        u_pred = None
        
        if i % recordPer == 0:
            error_vec = 0
            if graphPer != 0 and i % graphPer == 0:
                error_vec, u_pred = PINN.test(True)
            else:
                error_vec, u_pred = PINN.test(False)
            info.append([i, error_vec, loss.item(), loss_pde.item(), loss_bc.item()])
            
            # print(f"iter: {i}, PINN loss: {loss}, L2 relative error: {error_vec}, loss_pde: {loss_pde.item()}, loss_bc: {loss_bc.item()}")
            
            if i % savePer == 0:
            
                np.savetxt(f"{path}/history/Info_iter_{i}.csv", info)
                np.savetxt(f"{path}/prediction/PINNPrediction_iter_{i}.csv", u_pred)
            
    print("Training Time: ",time.time() - start_time)
    
    return PINN, info, samplePoints



def trainACGD(PINN, D, optimizer, max_iter, recordPer = 100, graphPer = 0, savePer = 1000, path = "", saveAt = [],
             miniBatch = False, batchSizeBC = 0, batchSizePDE = 0, lb = None, ub = None, u = None, f = None, trainBatchFor = 1):
#     Info = np.empty(((int)(max_iter / recordPer) + 1, 8)) #[i, error_vec, loss.item(), g_loss.item(), loss_pde.mean().item(), iter_num, iter_num_sum, hvp_count_sum]
    Info = []
    start_time = time.time()
    _, xy_bc, u_bc, xy_inside, f_xy = trainingData(lb, ub, batchSizeBC, batchSizePDE, u, f)
    iter_num_sum = 0
    for i in range(max_iter):
        optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
        
        loss1 = 0
        loss2 = 0
        
        if miniBatch: 
            if i % trainBatchFor == 0:
                _, xy_bc, u_bc, xy_inside, f_xy = trainingData(lb, ub, batchSizeBC, batchSizePDE, u, f)
                
            x_inside = xy_inside[:, [0]]
            y_inside = xy_inside[:, [1]]
            
            x_bc = xy_bc[:, [0]]
            y_bc = xy_bc[:, [1]]
            
            loss1 = D(x_inside, y_inside)[:,[0]] * PINN.test_PDE(x_inside, y_inside, f_xy)

            loss2 = D(x_bc, y_bc)[:,[1]] * (PINN(x_bc, y_bc) - u_bc)
        else:
            loss1 = D(PINN.x_inside_train, PINN.y_inside_train)[:,[0]] * PINN.test_PDE()

            loss2 = D(PINN.x_bc, PINN.y_bc)[:,[1]] * (PINN(PINN.x_bc, PINN.y_bc) - PINN.u_bc)

        loss = loss1.mean() +  loss2.mean()
        
        optimizer.step(loss, 0)
        iter_num  = optimizer.get_info()["iter_num"]
        iter_num_sum += iter_num
        u_pred = None
        if i % recordPer == 0:
            g_loss, loss_bc, loss_pde = PINN.loss()
            error_vec = 0
            if graphPer != 0 and i % graphPer == 0:
                error_vec, u_pred = PINN.test(True)
            else:
                error_vec, u_pred = PINN.test(False)
                
            # print('Epoch :{}, PINN error: {}, Total Loss: {}, loss_real: {}, loss_fake: {}'.format(e, error_vec, loss.item(), loss_real.item(), loss_fake.item()))
            # print('Epoch :{}, PINN loss: {}, relative error: {}, loss1: {}, loss2: {}, total loss: {}, pde_loss: {}, iter_num: {}, cumulative iter_num: {}'.format(
            #     i,  g_loss.item(), error_vec, loss1.mean().item(), loss2.mean().item(), loss.item(), loss_pde.item(), iter_num, iter_num_sum))
            Info.append([i, error_vec, loss.item(), g_loss.item(), loss_pde.item(), loss1.mean().item(), loss2.mean().item(), iter_num_sum])
            
        if i % savePer == 0 or i in saveAt:  
            D_output_save = D(PINN.x_test, PINN.y_test).cpu().detach().numpy()[:,[0]]
            
            np.savetxt(f"{path}/history/ACGDInfo_iter_{i}.csv", Info)
            np.savetxt(f"{path}/prediction/PINNPrediction_iter_{i}.csv", u_pred)
            np.savetxt(f"{path}/prediction/Discriminator_Prediction_iter_{i}.csv", D_output_save)
            torch.save({
                "PINN_state_dict": PINN.state_dict(),
                "Discriminator_state_dict": D.state_dict(),
                "ACGD_optimizer_state_dict" : optimizer.state_dict(),
            }, f"{path}/models/ACGD_models_iter_{i}.pt")
    print(f"Training Time: {time.time() - start_time}")
    return np.array(Info)

def trainBCGD(PINN, D, optimizer, max_iter, recordPer = 100, graphPer = 500,
             miniBatch = False, batchSizeBC = 0, batchSizePDE = 0, lb = None, ub = None, u = None, f = None, trainBatchFor = 1):
    # Info = np.empty(((int)(max_iter / recordPer), 6)) #[i, error_vec, loss.item(), g_loss.item(), loss_pde.mean().item()]
    Info = []
    start_time = time.time()
    _, xy_bc, u_bc, xy_inside, f_xy = trainingData(lb, ub, batchSizeBC, batchSizePDE, u, f)
    
    for i in range(max_iter):
        optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
        
        loss1 = 0
        loss2 = 0
        
        if miniBatch: 
            if i % trainBatchFor == 0:
                _, xy_bc, u_bc, xy_inside, f_xy = trainingData(lb, ub, batchSizeBC, batchSizePDE, u, f)
                
            x_inside = xy_inside[:, [0]]
            y_inside = xy_inside[:, [1]]
            
            x_bc = xy_bc[:, [0]]
            y_bc = xy_bc[:, [1]]
            
            loss1 = D(x_inside, y_inside)[:,[0]] * PINN.test_PDE(x_inside, y_inside, f_xy)

            loss2 = D(x_bc, y_bc)[:,[1]] * (PINN(x_bc, y_bc) - u_bc)
        else:
            loss1 = D(PINN.x_inside_train, PINN.y_inside_train)[:,[0]] * PINN.test_PDE()

            loss2 = D(PINN.x_bc, PINN.y_bc)[:,[1]] * (PINN(PINN.x_bc, PINN.y_bc) - PINN.u_bc)

        loss = loss1.mean() + loss2.mean()

        optimizer.step(loss, 0)
        if i % recordPer == 0:
            g_loss, loss_bc, loss_pde = PINN.loss()
            error_vec = 0
            if graphPer != 0 and i % graphPer == 0:
                error_vec, _ = PINN.test(True)
            else:
                error_vec, _ = PINN.test(False)
                
            iter_num  = optimizer.get_info()["iter_num"]

            # print('Epoch :{}, PINN error: {}, Total Loss: {}, loss_real: {}, loss_fake: {}'.format(e, error_vec, loss.item(), loss_real.item(), loss_fake.item()))
            # print('Epoch :{}, PINN loss: {}, relative error: {}, loss1: {}, loss2: {}, total loss: {}, pde_loss: {}, iter_num: {}'.format(
            #     i,  g_loss.item(), error_vec, loss1.mean().item(), loss2.mean().item(), loss.item(), loss_pde.item(), iter_num))
            Info.append([i, error_vec, loss.item(), g_loss.item(), loss_pde.item(), iter_num])
    print(f"Training Time: {time.time() - start_time}")
    return Info