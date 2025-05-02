import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
import numpy as np
import math
import matplotlib.pyplot as plt
from plot_results import plot_trajectories, plot_initial_target_dis_2d, plot_loss, plot_initial_target_dis_1d

'''
Flow matching solver for stochatic averaged systems
'''
print('CUDA runtime version', torch.version.cuda)
print('CUDA available', torch.cuda.is_available())
class FlowMatchingSolver:
    def __init__(self,
                 A_fn,
                 B_fn,
                 mu0,       # initial distribution
                 muf,       # target distribution
                 nx,        # dimension of state
                 nu,        # dimension of control input
                 epsilon=1e-2,
                 tf=1.0,
                 time_steps=100, # time steps for [0, tf]
                 theta_samples=50, # number of samples for the perturbed parameter theta
                 device='cpu'):  # device can be gpu
        self.A_fn = A_fn
        self.B_fn = B_fn
        self.mu0 = mu0
        self.muf = muf
        self.nx = nx
        self.nu = nu
        self.epsilon = epsilon
        self.tf = tf   # terminal time
        self.Nt = time_steps
        self.dt = tf / time_steps
        self.t_grid = torch.linspace(0, tf, time_steps+1, device=device)
        self.device = device
        #self.thetas = [torch.rand(()).item() for _ in range(theta_samples)]
        self.thetas = torch.linspace(0,1,theta_samples)

    def compute_Phi(self, t, tau):
        '''
        Phi(t, tau)
        '''
        A = torch.stack([self.A_fn(theta) for theta in self.thetas])
        B = torch.stack([self.B_fn(theta) for theta in self.thetas])
        Phi = torch.matrix_exp(A*(t-tau))@B
        return Phi.mean(dim=0)

    def compute_G(self, t):
        '''
        G(tf, t)
        '''
        G = torch.zeros(self.nx, self.nx, device=self.device)
        for tau in self.t_grid:
            if tau >= t:
                Phi = self.compute_Phi(self.tf, tau)
                G += Phi @ Phi.T * self.dt
        return G

    def sample_pairs(self, N):
        x0 = self.mu0.sample((N,))
        xf = self.muf.sample((N,))
        return x0.to(self.device), xf.to(self.device)

    def compute_control(self, x0, xf):
        # returns u_z trajectories: shape [N, Nt+1, nu]
        N, _ = x0.shape
        u_z = torch.zeros(N, self.Nt+1, self.nu, device=self.device)
        x_z = torch.zeros(N, self.Nt+1, self.nx, device=self.device)
        dW = torch.randn(N, self.Nt, nu, device=self.device) * math.sqrt(self.dt)
        feature_noise = torch.zeros(N, self.Nt+1, self.nx, device=self.device)
        for k, t in enumerate(self.t_grid): # for every time step
            Phi_tf_t = self.compute_Phi(self.tf, t)
            G_tf_0 = self.compute_G(0)
            Ginv = torch.linalg.inv(G_tf_0 + 1e-6 * torch.eye(nu, device=self.device))
            integral_term = torch.zeros(N, nu, device=self.device)
            Phi_t_tau_intergrand = torch.zeros(N, self.nx, device=self.device)
            for m, tau in enumerate(self.t_grid[:k]):
                Phi_tf_tau = self.compute_Phi(self.tf, tau)
                G_tf_tau = self.compute_G(tau)
                G_tf_tau_inv = torch.linalg.inv(G_tf_tau+1e-6*torch.eye(nu, device=self.device))
                integrand = Phi_tf_t.T @ G_tf_tau_inv @ Phi_tf_tau
                integral_term += dW[:, m]@integrand
                Phi_t_tau = self.compute_Phi(t, tau)
                Phi_t_tau_intergrand += dW[:, m]@Phi_t_tau.T*math.sqrt(self.epsilon)
            feature_noise[:, k] = Phi_t_tau_intergrand
            term1 = -math.sqrt(self.epsilon) * integral_term
            M_t = self.M(self.tf)
            term2 = (xf - x0.matmul(M_t.T))@Ginv.T@Phi_tf_t
            u_z[:, k] = term1 + term2
            M_t = self.M(t)
            u = u_z[:,k]
            control_integral_term = torch.zeros([N, nx], device=self.device)
            for m, tau in enumerate(self.t_grid[:k]):
                Phi_t_tau = self.compute_Phi(t, tau)
                control_integral_term += (Phi_t_tau @ (u.T)).T * self.dt
            x_z[:, k] = (M_t @ x0.T).T + control_integral_term + feature_noise[:,k]

        return u_z, feature_noise, x_z
        
    def M(self, t):
        M_integrand = torch.stack([self.A_fn(theta)*t for theta in self.thetas])
        M_integrand = torch.matrix_exp(M_integrand)
        return M_integrand.mean(dim=0)
    
    def build_dataset(self, x0, u_z, noise):
        N, T, nu = u_z.shape
        x0 = x0.unsqueeze(1).repeat(1, T, 1)
        t_all = self.t_grid.view(1, T, 1).repeat(N,1,1)
        X = torch.cat([x0, t_all,noise], dim=2)
        return X, u_z

    class LSTMRegressor(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out)

    class MLPRegressor(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
            super().__init__()
            layers = []
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            u = self.net(x)
            return u.squeeze(1)

    def train_control_law(self, X, Y,
                          hidden_dim=128,
                          lr=1e-3,
                          batch_size=128,
                          epochs=1000,
                          model_type='LSTM'):
        input_dim = X.size(2)
        output_dim = Y.size(2)
        if model_type == 'LSTM':
            model = self.LSTMRegressor(input_dim, hidden_dim, output_dim).to(self.device)
        elif model_type == 'MLP':
            model = self.MLPRegressor(input_dim, hidden_dim, output_dim, num_layers=2).to(self.device)
        else:
            print("Please choose from LSTM and MLP")

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model.train()
        epoch_loss = []
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            #print(f"Epoch {epoch+1}/{epochs}, Average loss: {total_loss/len(dataset):.6f}")
            epoch_loss.append([epoch, total_loss/len(dataset)])
        self.control_model = model
        return epoch_loss

    def simulate_x(self, x0):
        N, nx = x0.shape
        x = x0.clone()
        traj = torch.zeros(N, self.Nt+1, nx, device=self.device)
        traj[:, 0, :] = x
        dW = torch.randn(N, self.Nt, self.nu, device=self.device)*math.sqrt(self.dt)
        for k, t in enumerate(self.t_grid):
            noise_integral_term = torch.zeros([N, nx], device=self.device)
            for m, tau in enumerate(self.t_grid[:k]):
                Phi_t_tau = self.compute_Phi(t, tau)
                noise_integral_term += (Phi_t_tau @ dW[:, m].T).T
            noise_term = math.sqrt(self.epsilon)*noise_integral_term
            features = torch.cat([x0, t.unsqueeze(0).repeat(N,1), noise_term], dim=1)
            u = self.control_model(features.unsqueeze(1)).squeeze(1)
            M_t = self.M(t)
            control_integral_term = torch.zeros([N, nx], device=self.device)
            for m, tau in enumerate(self.t_grid[:k]):
                Phi_t_tau = self.compute_Phi(t, tau)
                control_integral_term += (Phi_t_tau@(u.T)).T*self.dt
            x = (M_t@x0.T).T + control_integral_term + noise_term
            traj[:, k, :] = x
        return traj, self.t_grid

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sys_name = 1  # test different systems, 0: example 1, 1: example 2
    if sys_name == 0:
        from Sys1 import sys, nx, nu, A_fn, B_fn
    else:
        from Sys2 import sys, nx, nu, A_fn, B_fn

    mu0 = MultivariateNormal(torch.zeros(nx), torch.eye(nx)) #initial distribution
    muf = MultivariateNormal(torch.ones(nx)*6, torch.eye(nx)) # target distribution

    print(f"using device {device}")
    # algorithm part
    print("Initialize solver")
    solver = FlowMatchingSolver(A_fn, B_fn, mu0, muf, nx, nu, epsilon=0.1, tf=1.0, time_steps=10, device=device)
    print("Sample from initial and target distribution")
    x0_samples, xf_samples = solver.sample_pairs(N=1000)
    print("Compute the control")
    u_z, noise, GT_traj = solver.compute_control(x0_samples, xf_samples)
    print("Prepare for the data set")
    X, Y = solver.build_dataset(x0_samples, u_z, noise)
    print("Start training")
    loss = solver.train_control_law(X, Y, model_type='MLP') # change it to MLP if you want to use MLP
    print("Test and get the trajectory with learned control")
    traj, t_grid = solver.simulate_x(x0_samples)
    print("Simulation done. Start to plot graph")
    save_dir = f'./results/{sys}/'
    plot_trajectories(traj, t_grid, save_dir)
    if sys == 'Example2':
        plot_initial_target_dis_2d(x0_samples, xf_samples, traj, t_grid, save_dir)
    elif sys == 'Example1':
        plot_initial_target_dis_1d(x0_samples, xf_samples, traj, save_dir)
    plot_loss(loss, save_dir)
    print('Figure saved in the results folder')


