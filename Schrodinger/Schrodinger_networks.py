import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

class PINN_Schrodinger(nn.Module):
    # Initialize the class
    def __init__(self, x0, u0, v0, tb, X_f, layers, lb, ub):
        super(PINN_Schrodinger, self).__init__() #call __init__ from parent class 
        self.activation = nn.Tanh()
        self.iter = 0
        X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
        X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
        X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
        
        self.lb = lb
        self.ub = ub

        self.lb = torch.from_numpy(self.lb).to(device).float()
        self.ub = torch.from_numpy(self.ub).to(device).float()
               
        self.x0 = X0[:,0:1]
        self.t0 = X0[:,1:2]

        self.x0 = torch.from_numpy(self.x0).to(device).float()
        self.t0 = torch.from_numpy(self.t0).to(device).float()

        self.x_lb = X_lb[:,0:1]
        self.t_lb = X_lb[:,1:2]
        
        self.x_lb = torch.from_numpy(self.x_lb).to(device).float()
        self.t_lb = torch.from_numpy(self.t_lb).to(device).float()

        self.x_ub = X_ub[:,0:1]
        self.t_ub = X_ub[:,1:2]

        self.x_ub = torch.from_numpy(self.x_ub).to(device).float()
        self.t_ub = torch.from_numpy(self.t_ub).to(device).float()
        
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        
        self.x_f = torch.from_numpy(self.x_f).to(device).float()
        self.t_f = torch.from_numpy(self.t_f).to(device).float()

        
        self.u0 = u0
        self.v0 = v0
        
        self.u0 = torch.from_numpy(self.u0).to(device).float()
        self.v0 = torch.from_numpy(self.v0).to(device).float()

        
        
        # Initialize NNs
        self.hiddenLayers = [nn.Linear(layers[0], layers[2])] + [nn.Linear(layers[2], layers[2]) for i in range(layers[1] - 1)] + [nn.Linear(layers[2], layers[3])]
        self.linears = nn.ModuleList(self.hiddenLayers)
        self.layers = layers
        
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)

        for i in range(len(layers)):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)

              
    def forward(self, x, t):
        
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x).to(device).float()
        if torch.is_tensor(t) != True:         
            t = torch.from_numpy(t).to(device).float()
        
        a = torch.cat((x, t), 1)
        # print(a.shape)
 
 
        for i in range(len(self.linears)-1):
            
            z = self.linears[i](a)
                        
            a = self.activation(z)
            
        a = self.linears[-1](a)
        
        return a
    
        
    def loss(self):
        criterion = nn.MSELoss()

        # loss 0
        self.x0.requires_grad = True
        self.t0.requires_grad = True
        out0 = self.forward(self.x0, self.t0)

        u0_pred, v0_pred = out0[:, [0]], out0[:, [1]]

        loss0 = criterion(u0_pred, self.u0) + criterion(v0_pred, self.v0)

        u_lb_pred, u_ub_pred, v_lb_pred, v_ub_pred, u_x_lb_pred, u_x_ub_pred, v_x_lb_pred, v_x_ub_pred = self.residual_bc()
        lossb = criterion(u_lb_pred, u_ub_pred) + criterion(v_lb_pred, v_ub_pred)

        f_u, f_v = self.residual_f()
        lossf = criterion(f_u, torch.zeros_like(f_u)) + criterion(f_v, torch.zeros_like(f_v))

        total_loss = loss0 + lossb + lossf

        return total_loss, loss0, lossb, lossf
        # lossb
    def residual_bc(self):
        self.x_lb.requires_grad = True
        self.t_lb.requires_grad = True

        outlb = self.forward(self.x_lb, self.t_lb) # lower boundary output


        u_lb_pred, v_lb_pred = outlb[:, [0]], outlb[:, [1]]

        u_x_lb_pred = autograd.grad(u_lb_pred, self.x_lb, torch.ones([self.x_lb.shape[0], 1]).to(device), retain_graph = True, create_graph = True)[0]
        v_x_lb_pred = autograd.grad(v_lb_pred, self.x_lb, torch.ones([self.x_lb.shape[0], 1]).to(device), retain_graph = True, create_graph = True)[0]


        self.x_ub.requires_grad = True
        self.t_ub.requires_grad = True

        outub = self.forward(self.x_ub, self.t_ub) # upper boundary output

        u_ub_pred, v_ub_pred = outub[:, [0]], outub[:, [1]]

        u_x_ub_pred = autograd.grad(u_ub_pred, self.x_ub, torch.ones([self.x_ub.shape[0], 1]).to(device), retain_graph = True, create_graph = True)[0]
        v_x_ub_pred = autograd.grad(v_ub_pred, self.x_ub, torch.ones([self.x_ub.shape[0], 1]).to(device), retain_graph = True, create_graph = True)[0]

        return u_lb_pred, u_ub_pred, v_lb_pred, v_ub_pred, u_x_lb_pred, u_x_ub_pred, v_x_lb_pred, v_x_ub_pred


    def residual_f(self):
        self.x_f.requires_grad = True
        self.t_f.requires_grad = True
        outf = self.forward(self.x_f, self.t_f)


        u_f_pred, v_f_pred = outf[:, [0]], outf[:, [1]]


        u_f_x_pred = autograd.grad(u_f_pred, self.x_f, torch.ones([self.x_f.shape[0], 1]).to(device), retain_graph = True, create_graph = True)[0]
        v_f_x_pred = autograd.grad(v_f_pred, self.x_f, torch.ones([self.x_f.shape[0], 1]).to(device), retain_graph = True, create_graph = True)[0]

        
        u_f_t_pred = autograd.grad(u_f_pred, self.t_f, torch.ones([self.t_f.shape[0], 1]).to(device), retain_graph = True, create_graph = True)[0]
        u_f_xx_pred = autograd.grad(u_f_x_pred, self.x_f, torch.ones([self.x_f.shape[0], 1]).to(device), retain_graph = True, create_graph = True)[0] # tf.gradients(u_x, x)[0]
        
        v_f_t_pred = autograd.grad(v_f_pred, self.t_f, torch.ones([self.t_f.shape[0], 1]).to(device), retain_graph = True, create_graph = True)[0]
        v_f_xx_pred = autograd.grad(v_f_x_pred, self.x_f, torch.ones([self.x_f.shape[0], 1]).to(device), retain_graph = True, create_graph = True)[0]
        
        f_u = u_f_t_pred + 0.5*v_f_xx_pred + (u_f_pred ** 2 + v_f_pred ** 2) * v_f_pred
        f_v = v_f_t_pred - 0.5*u_f_xx_pred - (u_f_pred ** 2 + v_f_pred ** 2) * u_f_pred

        return f_u, f_v

    

    def test(self, X_star, u_star, v_star):
        x = X_star[:, [0]]
        t = X_star[:, [1]]

        output = self.forward(x, t)
        u_pred = output[:, [0]].cpu().detach().numpy()
        v_pred = output[:, [1]].cpu().detach().numpy()
        
        h_pred = np.sqrt(u_pred**2 + v_pred**2)
        h_star = np.sqrt(u_star**2 + v_star**2)

        error_u = np.linalg.norm(u_pred - u_star, 2) / np.linalg.norm(u_star, 2)
        error_v = np.linalg.norm(v_pred - v_star, 2) / np.linalg.norm(v_star, 2)
        error_h = np.linalg.norm(h_pred - h_star, 2) / np.linalg.norm(h_star, 2)
        

        return error_u, error_v, error_h, u_pred, v_pred


class Discriminator(nn.Module):
    def __init__(self, layers):
        super(Discriminator, self).__init__()
        self.hiddenLayers = [nn.Linear(layers[0], layers[2])] + [nn.ReLU(), nn.Linear(layers[2], layers[2])] * (layers[1] - 1)+ [nn.ReLU(), nn.Linear(layers[2], layers[3])]
        self.linears = nn.ModuleList(self.hiddenLayers)

    def forward(self, input):
        # temp = torch.cat((x, t), 1)
        temp = input
        for layer in self.linears:
            temp = layer(temp)
        return temp