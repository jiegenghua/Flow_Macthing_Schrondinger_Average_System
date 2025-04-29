import numpy as np
import matplotlib.pyplot as plt
import torch
import os
def plot_trajectories(traj, t_grid, log_dir):
    os.makedirs(log_dir, exist_ok=True)

    if torch.is_tensor(traj):
        traj = traj.detach().cpu().numpy()
        t_grid = t_grid.detach().cpu().numpy()

    n_samples, _, state_dim = traj.shape
    for dim in range(state_dim):
        plt.figure(figsize=(8,6))
        for sample in range(n_samples):
            plt.plot(t_grid, traj[sample,:, dim])
            plt.xlabel(r'$t$')
            plt.ylabel(fr'$x_{dim}$')
            plt.savefig(log_dir+f'x{dim}.jpg')


def plot_initial_target_dis_2d(x0, xf, traj,t_grid, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    if torch.is_tensor(traj):
        x0 =x0.detach().cpu().numpy()
        xf = xf.detach().cpu().numpy()
        traj = traj.detach().cpu().numpy()
        t_grid = t_grid.detach().cpu().numpy()

    plt.figure(figsize=(8,6))
    N, nx = x0.shape
    for i in range(N):
        plt.plot(traj[i,:,0], traj[i,:,1], color='m', alpha=0.1)
        
    plt.scatter(x0[:, 0], x0[:, 1], color='b', label='initial')
    plt.scatter(xf[:,0],  xf[:,1], color='r', label='target')
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.legend(loc=2)
    plt.xlabel(r'$x_t(1)$')
    plt.ylabel(r'$x_t(2)$')
    plt.savefig(log_dir+f'traj.jpg')