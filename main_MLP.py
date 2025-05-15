import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
import numpy as np
import math
import matplotlib.pyplot as plt
from plot_results import plot_trajectories, plot_initial_target_dis_2d, plot_loss, plot_initial_target_dis_1d, W2_2d, W2_1d
import os
import sys
from utils import sample_2G, sample_4G, Circle, HalfMoon

'''
Flow matching solver for stochatic averaged systems
'''


class FlowMatchingSolver:
    def __init__(self,
                 A_fn,
                 B_fn,
                 nx,  # dimension of state
                 nu,  # dimension of control input
                 epsilon=1e-2,
                 tf=1.0,
                 time_steps=100,  # time steps for [0, tf]
                 theta_samples=50,  # number of samples for the perturbed parameter theta
                 device='cpu'):  # device can be gpu
        self.A_fn = A_fn
        self.B_fn = B_fn
        self.nx = nx
        self.nu = nu
        self.epsilon = epsilon
        self.tf = tf  # terminal time
        self.Nt = time_steps
        self.dt = tf / time_steps
        self.t_grid = torch.linspace(0, tf, time_steps + 1, device=device)
        self.device = device
        # self.thetas = [torch.rand(()).item() for _ in range(theta_samples)]
        self.thetas = torch.linspace(0, 1, theta_samples)

    def compute_Phi(self, t, tau):
        '''
        Phi(t, tau)
        '''
        A = torch.stack([self.A_fn(theta) for theta in self.thetas])
        B = torch.stack([self.B_fn(theta) for theta in self.thetas])
        Phi = torch.matrix_exp(A * (t - tau)) @ B
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

    def compute_control(self, x0, xf):
        # returns u_z trajectories: shape [N, Nt+1, nu]
        N, _ = x0.shape
        u_z = torch.zeros(N, self.Nt + 1, self.nu, device=self.device)
        x_z = torch.zeros(N, self.Nt + 1, self.nx, device=self.device)
        dW = torch.randn(N, self.Nt, nu, device=self.device) * math.sqrt(self.dt)
        feature_noise = torch.zeros(N, self.Nt + 1, self.nx, device=self.device)
        for k, t in enumerate(self.t_grid):  # for every time step
            Phi_tf_t = self.compute_Phi(self.tf, t)
            G_tf_0 = self.compute_G(0)
            Ginv = torch.linalg.inv(G_tf_0 + 1e-6 * torch.eye(nu, device=self.device))
            integral_term = torch.zeros(N, nu, device=self.device)
            Phi_t_tau_intergrand = torch.zeros(N, self.nx, device=self.device)
            for m, tau in enumerate(self.t_grid[:k]):
                Phi_tf_tau = self.compute_Phi(self.tf, tau)
                G_tf_tau = self.compute_G(tau)
                G_tf_tau_inv = torch.linalg.inv(G_tf_tau + 1e-6 * torch.eye(nu, device=self.device))
                integrand = Phi_tf_t.T @ G_tf_tau_inv @ Phi_tf_tau
                integral_term += dW[:, m] @ integrand
                Phi_t_tau = self.compute_Phi(t, tau)
                Phi_t_tau_intergrand += dW[:, m] @ Phi_t_tau.T * math.sqrt(self.epsilon)
            feature_noise[:, k] = Phi_t_tau_intergrand
            term1 = -math.sqrt(self.epsilon) * integral_term
            M_t = self.M(self.tf)
            term2 = (xf - x0.matmul(M_t.T)) @ Ginv.T @ Phi_tf_t
            u_z[:, k] = term1 + term2
            M_t = self.M(t)
            u = u_z[:, k]
            control_integral_term = torch.zeros([N, nx], device=self.device)
            for m, tau in enumerate(self.t_grid[:k]):
                Phi_t_tau = self.compute_Phi(t, tau)
                control_integral_term += (Phi_t_tau @ (u.T)).T * self.dt
            x_z[:, k] = (M_t @ x0.T).T + control_integral_term + feature_noise[:, k]

        return u_z, feature_noise, x_z

    def M(self, t):
        M_integrand = torch.stack([self.A_fn(theta) * t for theta in self.thetas])
        M_integrand = torch.matrix_exp(M_integrand)
        return M_integrand.mean(dim=0)

    def build_dataset(self, x0, u_z, noise):
        N, T, nu = u_z.shape
        # x0 = x0.unsqueeze(1).repeat(1, T, 1)
        t_all = self.t_grid.view(1, T, 1).repeat(N, 1, 1)
        X = torch.cat([x0, t_all, noise], dim=2)
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
            return u

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
            # print(f"Epoch {epoch+1}/{epochs}, Average loss: {total_loss/len(dataset):.6f}")
            epoch_loss.append([epoch, total_loss / len(dataset)])
        self.control_model = model
        return epoch_loss

    def simulate_x(self, x0):
        N, nx = x0.shape
        x = x0.clone()
        traj = torch.zeros(N, self.Nt + 1, nx, device=self.device)
        traj[:, 0, :] = x
        dW = torch.randn(N, self.Nt + 1, self.nu, device=self.device) * math.sqrt(self.dt)
        u_all = torch.zeros(N, self.Nt + 1, nu, device=self.device)
        u = torch.zeros(N, nu, device=self.device)

        for k, t in enumerate(self.t_grid):
            noise_integral_term = torch.zeros([N, nx], device=self.device)
            for m, tau in enumerate(self.t_grid[:k]):
                Phi_t_tau = self.compute_Phi(t, tau)
                noise_integral_term += (Phi_t_tau @ dW[:, m].T).T
            noise_term = math.sqrt(self.epsilon) * noise_integral_term
            features_seq = torch.cat([x, t.unsqueeze(0).repeat(N, 1), noise_term], dim=1)
            # feed the feature into the model
            u = self.control_model(features_seq)
            u_all[:, k, :] = u
            
            control_integral_term = torch.zeros([N, nx], device=self.device)
            for m, tau in enumerate(self.t_grid[:k]):
                Phi_t_tau = self.compute_Phi(t, tau)
                control_integral_term += (Phi_t_tau @ (u_all[:, m].T)).T * self.dt
            
            M_t = self.M(t)
            x = (M_t @ x0.T).T + control_integral_term + noise_term
            traj[:, k, :] = x
        
        return traj, u_all, self.t_grid


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if len(sys.argv) !=4:
        print("Please input the command as: python main_MLP.py 0 Gaussian Gaussian (2G, 4G, Circle)")
        sys.exit(1)
    sys_id = sys.argv[1] # example id
    init_dist = sys.argv[2] # initial distribution 
    target_dist = sys.argv[3] # target distribution

    N = 1000 # number of samples
    if sys_id == '0':
        from Sys1 import sys_name, nx, nu, A_fn, B_fn, mu0, muf
    else:
        from Sys2 import sys_name, nx, nu, A_fn, B_fn
    
    if init_dist == 'Gaussian':
        mu0 = MultivariateNormal(torch.ones(nx)*-1, torch.eye(nx))  # initial distribution
        x0_samples = mu0.sample((N,)).to(device)
    elif init_dist == '2G':
        x0_samples = sample_2G(N, nx, device).to(device)
    elif init_dist == '4G':
        x0_samples = sample_4G(N, nx, device).to(device)
    elif init_dist =='Circle' and nx==2:
        x0_samples = Circle(N, 2, device).to(device)
    elif init_dist == 'HalfMoon':
        x0_samples = HalfMoon(N, device).to(device)
    else:
        print("Please input a correct initial distribution (Gaussian, 2G, 4G, Circle, HalfMoon)")   
    
    if target_dist == 'Gaussian':
        muf = MultivariateNormal(torch.ones(nx) * 4, torch.eye(nx))  # target distribution
        xf_samples = muf.sample((N,)).to(device)
    elif target_dist == '2G':
        xf_samples = sample_2G(N, nx, device).to(device)
    elif target_dist == '4G':
        xf_samples = sample_4G(N, nx, device).to(device)
    elif target_dist=='Circle' and nx==2:
        xf_samples = Circle(N, 2, device).to(device)
    elif target_dist == 'HalfMoon' and nx==2:
        xf_samples = HalfMoon(N, device).to(device)
    else:
        print("Please input a correct target distribution (Gaussian, 2G, 4G, Circle, HalfMoon)")

    print(f"using device {device}")
    # algorithm part
    print("Initialize solver")
    solver = FlowMatchingSolver(A_fn, B_fn, nx, nu, epsilon=0.01, tf=1.0, time_steps=10, device=device)
    print("Compute the control")
    u, noise, GT_traj = solver.compute_control(x0_samples, xf_samples)
    print("Prepare for the data set")
    X, Y = solver.build_dataset(GT_traj, u, noise)
    print("Start training")
    loss = solver.train_control_law(X, Y, model_type='MLP')  # change it to MLP if you want to use MLP
    print("Test and get the trajectory with learned control")
    traj, u_z, t_grid = solver.simulate_x(x0_samples)
    print("Simulation done. Start to plot graph")
    save_dir = os.path.join('.', 'results', sys_name, init_dist, target_dist)
    print(f"The results will be saved in {save_dir}")
    plot_trajectories(traj, u, u_z, t_grid, save_dir)
    if sys_name == 'Example2':
        plot_initial_target_dis_2d(x0_samples, xf_samples, GT_traj, traj, t_grid, save_dir)
        W2_2d(x0_samples, xf_samples, GT_traj, traj, t_grid, save_dir)
    elif sys_name == 'Example1':
        plot_initial_target_dis_1d(x0_samples, xf_samples, GT_traj, traj, save_dir)
        W2_1d(x0_samples, xf_samples, GT_traj, traj, t_grid, save_dir)
    plot_loss(loss, save_dir)
    print('Figure saved in the results folder')


