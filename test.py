import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
import numpy as np
import math


def matrix_exponential(A, t):
    """Compute matrix exponential exp(A * t)."""
    return torch.matrix_exp(A * t)


def avg_over_theta(func, thetas):
    """Average a function of theta over given thetas."""
    values = [func(theta) for theta in thetas]
    return sum(values) / len(values)


class FlowMatchingSolver:
    def __init__(self,
                 A_fn,
                 B_fn,
                 mu0,       # torch.distributions
                 muf,       # torch.distributions
                 eps=1e-2,
                 T=1.0,
                 time_steps=100,
                 theta_samples=16,
                 device='cpu'):
        self.A_fn = A_fn
        self.B_fn = B_fn
        self.mu0 = mu0
        self.muf = muf
        self.eps = eps
        self.T = T
        self.Nt = time_steps
        self.dt = T / time_steps
        self.t_grid = torch.linspace(0, T, time_steps+1, device=device)
        self.device = device
        # pre-sample thetas for averaging
        self.thetas = [torch.rand(()).item() for _ in range(theta_samples)]

    def compute_Phi(self, t, tau):
        # Φ(t, τ) = ∫_0^1 exp(A(θ) (t-τ)) B(θ) dθ
        def integrand(theta):
            A = self.A_fn(theta)
            B = self.B_fn(theta)
            return matrix_exponential(A, (t - tau)) @ B
        return avg_over_theta(integrand, self.thetas)

    def compute_G(self, t):
        # G_{T,t} = ∫_t^T Φ(T, τ) Φ(T, τ)^T dτ
        G = torch.zeros_like(self.compute_Phi(self.T, self.t_grid[0]) @ self.compute_Phi(self.T, self.t_grid[0]).T)
        for tau in self.t_grid:
            if tau >= t:
                Phi = self.compute_Phi(self.T, tau)
                G += Phi @ Phi.T * self.dt
        return G

    def sample_pairs(self, N):
        x0 = self.mu0.sample((N,))
        xf = self.muf.sample((N,))
        return x0.to(self.device), xf.to(self.device)

    def compute_control_trajectories(self, x0, xf):
        # returns u_z trajectories: shape [N, Nt+1, dim]
        N, dim = x0.shape
        u_z = torch.zeros(N, self.Nt+1, dim, device=self.device)
        # Precompute G and its inverse for t=0
        # but actually G depends on t -> we compute per time step inside loop
        for i in range(N):
            # simulate Brownian increments dW
            dW = torch.randn(self.Nt, dim, device=self.device) * math.sqrt(self.dt)
            Wint = torch.zeros(self.Nt+1, dim, device=self.device)
            for k in range(1, self.Nt+1):
                Wint[k] = Wint[k-1] + dW[k-1]
            for k, t in enumerate(self.t_grid):
                # compute Phi and G
                Phi_Tt = self.compute_Phi(self.T, t)
                G_Tt = self.compute_G(t)
                Ginv = torch.linalg.inv(G_Tt + 1e-6 * torch.eye(dim, device=self.device))
                # first term: - sqrt(eps) ∫_0^t Φ(t, τ)^T G^{-1} Φ(T, τ) dW(τ)
                # approximate integral over history
                integral_term = torch.zeros(dim, device=self.device)
                for m, tau in enumerate(self.t_grid[:k]):
                    Phi_tt = self.compute_Phi(t, tau)
                    Phi_Ttau = self.compute_Phi(self.T, tau)
                    integrand = Phi_tt.T @ Ginv @ Phi_Ttau
                    integral_term += integrand @ dW[m]
                term1 = -math.sqrt(self.eps) * integral_term
                # second term: Φ(T, t)^T G^{-1} (xf - M(t)x0)
                # M(t) = ∫_0^1 exp(A(θ) t) dθ
                def M_integrand(theta):
                    A = self.A_fn(theta)
                    return matrix_exponential(A, t)
                M_t = avg_over_theta(M_integrand, self.thetas)
                term2 = Phi_Tt.T @ Ginv @ (xf[i] - M_t @ x0[i])
                u_z[i, k] = term1 + term2
        return u_z

    def build_dataset(self, x0, u_z):
        # features: [x0, t, noisy_obs]
        # noisy_obs = sqrt(eps) ∫_0^t Phi(t, τ) dW(τ)
        N, _, dim = u_z.shape
        X = []
        Y = []
        for i in range(N):
            Wint = torch.zeros_like(x0[i])
            for k, t in enumerate(self.t_grid):
                if k > 0:
                    dW = torch.randn(dim, device=self.device) * math.sqrt(self.dt)
                    # approximate sqrt(eps) ∫ Phi dW
                    Wint += torch.sqrt(torch.tensor(self.eps)) * (self.compute_Phi(t, self.t_grid[k-1]) @ dW)
                feat = torch.cat([x0[i], t.unsqueeze(0), Wint])
                X.append(feat)
                Y.append(u_z[i, k])
        return torch.stack(X), torch.stack(Y)

    class LSTMRegressor(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # x: [batch, seq=1, input_dim]
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    def train_control_law(self, X, Y,
                          hidden_dim=64,
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
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataset):.6f}")
        self.control_model = model

    def simulate(self, x0):
        # simulate controlled process x_eps(t)
        N, dim = x0.shape
        x = x0.clone()
        traj = torch.zeros(N, self.Nt+1, dim, device=self.device)
        traj[:, 0, :] = x
        for k, t in enumerate(self.t_grid[:-1]):
            # compute control u_epsilon via model
            features = torch.cat([x0, t.unsqueeze(0).repeat(N,1), torch.zeros(N, dim, device=self.device)], dim=1)
            u = self.control_model(features.unsqueeze(1))
            # update state: x <- M(dt)x + Phi * u * dt + sqrt(eps)dW
            # here approximate M(delta t) ~ I + A*dt
            A0 = self.A_fn(0.5)  # representative theta
            x = (torch.eye(dim, device=self.device) + A0 * self.dt) @ x.unsqueeze(-1)
            x = x.squeeze(-1) + (self.compute_Phi(t, t) @ u.T).T * self.dt
            x = x + torch.randn_like(x) * math.sqrt(self.eps * self.dt)
            traj[:, k+1, :] = x
        return traj

# Example usage:
if __name__ == "__main__":
    dim = 2
    A_fn = lambda theta: torch.tensor([[-theta, 0.0], [0.0, -theta]])
    B_fn = lambda theta: torch.eye(dim)
    mu0 = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    muf = MultivariateNormal(torch.ones(dim)*2, torch.eye(dim))
    solver = FlowMatchingSolver(A_fn, B_fn, mu0, muf, eps=0.1, T=1.0, time_steps=50, device='cpu')
    x0_samples, xf_samples = solver.sample_pairs(N=100)
    u_z = solver.compute_control_trajectories(x0_samples, xf_samples)
    X, Y = solver.build_dataset(x0_samples, u_z)
    solver.train_control_law(X, Y)
    traj = solver.simulate(x0_samples)
    print("Simulation done.")
