import numpy as np
import matplotlib.pyplot as plt
import torch
import os
def plot_trajectories(traj, t_grid):
    log_dir = './results/'
    os.makedirs(log_dir, exist_ok=True)

    if torch.is_tensor(traj):
        traj = traj.detach().cpu().numpy()
        t_grid = t_grid.detach().cpu().numpy()

    n_samples, time_steps, state_dim = traj.shape
    for dim in range(state_dim):
        plt.figure()
        for sample in range(n_samples):
            plt.plot(t_grid, traj[sample,:, dim])
            plt.xlabel(r'$t$')
            plt.ylabel(fr'$x_{dim}$')
            plt.savefig(log_dir+f'x{dim}.jpg')