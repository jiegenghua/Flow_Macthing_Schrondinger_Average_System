import torch
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn

class ControlLSTM(nn.Module):
    def __init__(self, state_dim, hidden_dim, control_dim, num_layers=2):
        super().__init__()
        # input_dim = state_dim + 1 (for time)
        self.lstm = nn.LSTM(state_dim, hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        # project hidden state to control
        self.head = nn.Linear(hidden_dim, control_dim)

    def forward(self, x_seq):
        """
        x_seq: (B, S, state_dim)
        t_seq: (B, S,    1     )
        returns: u_pred of shape (B, S, control_dim)
        """
        # concatenate along feature axis
        inp = x_seq      # (B, S, state_dim+1)
        out, _ = self.lstm(inp)                      # (B, S, hidden_dim)
        return self.head(out)                        # (B, S, control_dim)

# -------------------
# Example usage
# -------------------
if __name__ == "__main__":
    # 1) Generate toy data
    device = 'cpu'
    t0, tf, T = 0.0, 1.0, 50
    dt = (tf - t0) / (T - 1)
    t_grid = torch.linspace(t0, tf, T)  # (T,)
    N, nx, nu = 1000, 1, 1
    x0 = torch.zeros(N, nx)
    u_data = torch.ones(N, T, nu)*0.1
    x_data = torch.zeros(N, T, nx)
    x_data[:, 0, :] = x0
    for k in range(1, T):
        x_data[:, k, :] = x_data[:, k - 1, :] + dt * u_data[:, k - 1, :]

    model = ControlLSTM(nx, hidden_dim=64, control_dim=nu).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    batch_size = 32
    for epoch in range(1, 50):
        model.train()
        total_loss = 0.0
        B,_,_ = x_data.shape
        perm = torch.randperm(B)
        import numpy as np
        for i in range(0, B):
            idx = perm[i:i+batch_size]
            xb = x_data[idx]
            ub = u_data[idx]
            #tb = t_grid.unsqueeze(0).expand(len(idx),-1,-1)
            #print("ahahaha", np.shape(xb), np.shape(ub), np.shape(tb))
            #inp_b = torch.cat([xb, tb], dim=2)
            inp_b = xb
            pred = model(inp_b)
            # if predicting only last step:
            # pred = pred[:, -1, :]                    # (B, nu)
            loss = loss_fn(pred, ub)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        print(f"Epoch {epoch}, Loss = {total_loss / B:.6f}")

    import matplotlib.pyplot as plt
    x0 = torch.zeros(N, nx)
    u_data = torch.randn(N, T, nu)
    x_data_new = torch.zeros(N, T, nx)
    x_data_new[:, 0, :] = x0
    for k in range(1, T):
        u = model(x0).detach().numpy()
        x_next = x0 + dt * u
        x_data_new[:, k, :] = x_next
        x0 = x_next[:]

    plt.figure(1)
    plt.plot(t_grid, x_data[0, :, :], label='from real data')
    plt.plot(t_grid, x_data_new[0, :, :], label='from learned model')
    plt.legend()
    plt.show()
