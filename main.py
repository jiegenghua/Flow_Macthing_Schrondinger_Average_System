import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
import numpy as np
import math
import matplotlib.pyplot as plt
from plot_results import plot_trajectories
'''
Flow matching solver for stochatic averaged systems
'''

def avg_over_theta(func, thetas):
    values = [func(theta) for theta in thetas]
    return sum(values) / len(values)


class FlowMatchingSolver:
    def __init__(self,
                 A_fn,
                 B_fn,
                 mu0,       # initial distribution
                 muf,       # target distribution
                 epsilon=1e-2,
                 tf=1.0,
                 time_steps=100, # time steps for [0, tf]
                 theta_samples=16, # number of samples for the perturbed parameter theta
                 device='cpu'):  # device can be gpu
        self.A_fn = A_fn
        self.B_fn = B_fn
        self.mu0 = mu0
        self.muf = muf
        self.epsilon = epsilon
        self.tf = tf   # terminal time
        self.Nt = time_steps
        self.dt = tf / time_steps
        self.t_grid = torch.linspace(0, tf, time_steps+1, device=device)
        self.device = device
        self.thetas = [torch.rand(()).item() for _ in range(theta_samples)]

    def compute_Phi(self, t, tau):
        '''
        Phi(t, tau)
        '''
        def integrand(theta):
            A = self.A_fn(theta)
            B = self.B_fn(theta)
            return torch.matrix_exp(A * (t-tau)) @ B
        return avg_over_theta(integrand, self.thetas)

    def compute_G(self, t):
        '''
        G(tf, t)
        '''
        G = torch.zeros_like(self.compute_Phi(self.tf, self.t_grid[0]) @ self.compute_Phi(self.tf, self.t_grid[0]).T)
        for tau in self.t_grid:
            if tau >= t:
                Phi = self.compute_Phi(self.tf, tau)
                G += Phi @ Phi.T * self.dt
        return G

    def sample_pairs(self, N):
        x0 = self.mu0.sample((N,))
        xf = self.muf.sample((N,))
        return x0.to(self.device), xf.to(self.device)

    def compute_control_trajectories(self, x0, xf):
        # returns u_z trajectories: shape [N, Nt+1, dim]
        N, nu = x0.shape
        u_z = torch.zeros(N, self.Nt+1, nu, device=self.device)
        for i in range(N): # for every sample
            dW = torch.randn(self.Nt, nu, device=self.device) * math.sqrt(self.dt)
            Wint = torch.zeros(self.Nt+1, nu, device=self.device)
            for k in range(1, self.Nt+1):
                Wint[k] = Wint[k-1] + dW[k-1]
            for k, t in enumerate(self.t_grid): # for every time step
                Phi_tf_t = self.compute_Phi(self.tf, t)
                G_tf_0 = self.compute_G(0)
                Ginv = torch.linalg.inv(G_tf_0 + 1e-6 * torch.eye(nu, device=self.device))
                integral_term = torch.zeros(nu, device=self.device)
                for m, tau in enumerate(self.t_grid[:k]):
                    Phi_tf_tau = self.compute_Phi(self.tf, tau)
                    G_tf_tau = self.compute_G(tau)
                    G_tf_tau_inv = torch.linalg.inv(G_tf_tau+1e-6*torch.eye(nu, device=self.device))
                    integrand = Phi_tf_t.T @ G_tf_tau_inv @ Phi_tf_tau
                    integral_term += integrand @ dW[m]
                term1 = -math.sqrt(self.epsilon) * integral_term

                def M_integrand(theta):
                    A = self.A_fn(theta)
                    return torch.matrix_exp(A*self.tf)
                M_t = avg_over_theta(M_integrand, self.thetas)
                term2 = Phi_tf_t.T @ Ginv @ (xf[i] - M_t @ x0[i])
                u_z[i, k] = term1 + term2
        return u_z
        
    def M(self, t):
        A = self.A_fn(0)
        M_integrand = torch.zeros_like(A)
        for theta in self.thetas:
            M_integrand += torch.matrix_exp(self.A_fn(theta)*t)
        return torch.matrix_exp(A*t)
    
    def build_dataset(self, x0, u_z):
        N, _, nu = u_z.shape
        X = []
        Y = []
        for i in range(N):
            Wint = torch.zeros_like(x0[i])
            for k, t in enumerate(self.t_grid):
                if k > 0:
                    dW = torch.randn(nu, device=self.device) * math.sqrt(self.dt)
                    Wint += torch.sqrt(torch.tensor(self.epsilon)) * (self.compute_Phi(t, self.t_grid[k-1]) @ dW)
                input = torch.cat([x0[i], t.unsqueeze(0), Wint])
                X.append(input)
                Y.append(u_z[i, k])
        return torch.stack(X), torch.stack(Y)

    class LSTMRegressor(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    def train_control_law(self, X, Y,
                          hidden_dim=12,
                          lr=1e-3,
                          batch_size=128,
                          epochs=20):
        input_dim = X.size(1)
        output_dim = Y.size(1)
        model = self.LSTMRegressor(input_dim, hidden_dim, output_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        dataset = torch.utils.data.TensorDataset(X.unsqueeze(1), Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model.train()
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
            print(f"Epoch {epoch+1}/{epochs}, Average loss: {total_loss/len(dataset):.6f}")
        self.control_model = model

    def simulate_x(self, x0):
        N, nx = x0.shape
        x = x0.clone()
        traj = torch.zeros(N, self.Nt+1, nx, device=self.device)
        traj[:, 0, :] = x
        dW = torch.randn(self.Nt, nx, device=self.device)
        for k, t in enumerate(self.t_grid[1:]): 
            noise_integral_term = torch.zeros([N, nx], device=self.device)
            for m, tau in enumerate(self.t_grid[:k+1]):
                Phi_t_tau = self.compute_Phi(t, tau)
                noise_integral_term += Phi_t_tau @ dW[m]*self.dt
            noise_term = math.sqrt(self.epsilon)*noise_integral_term
            features = torch.cat([x0, t.unsqueeze(0).repeat(N,1), noise_term], dim=1)
            u = self.control_model(features.unsqueeze(1))
            M_t = self.M(t)
            control_integral_term = torch.zeros([N, nx], device=self.device)
            for m, tau in enumerate(self.t_grid[:k+1]):
                Phi_t_tau = self.compute_Phi(t, tau)
                control_integral_term += (Phi_t_tau@(u.T)).T*self.dt
            x = (M_t@x0.T).T + control_integral_term + noise_term
            traj[:, k+1, :] = x
        return traj, self.t_grid

if __name__ == "__main__":
    '''
    example 2
    '''
    nx = 2   # dimension of the state
    nu = 2
    A_fn = lambda theta: torch.tensor([[-theta, 0.0], [0.0, -theta]])
    B_fn = lambda theta: torch.eye(nx)

    '''
    example 1
    '''
    #nx = 1 # dimension of x
    #nu =1  # dimension of u
    #A_fn = lambda theta: torch.tensor([[0.0, -theta], [theta, 0]])
    #B_fn = lambda theta: torch.eye(nx)

    mu0 = MultivariateNormal(torch.zeros(nx), torch.eye(nx)) #initial distribution
    muf = MultivariateNormal(torch.ones(nx)*2, torch.eye(nx)) # target distribution

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print(f"using device {device}")
    # algorithm part
    print("Initialize solver")
    solver = FlowMatchingSolver(A_fn, B_fn, mu0, muf, epsilon=0.1, tf=1.0, time_steps=10, device='cpu')
    print("Sample from initial and target distribution")
    x0_samples, xf_samples = solver.sample_pairs(N=100)
    print("Compute the control")
    u_z = solver.compute_control_trajectories(x0_samples, xf_samples)
    print("Prepare for the data set")
    X, Y = solver.build_dataset(x0_samples, u_z)
    print("Start LSTM training")
    solver.train_control_law(X, Y)
    print("Test and get the trajectory with learned control")
    n_sample = 100
    x0_test = torch.zeros([n_sample, nx])
    traj, t_grid = solver.simulate_x(x0_test)
    print("Simulation done. Start to plot graph")
    plot_trajectories(traj, t_grid)
    print('Figure saved in the results folder')


