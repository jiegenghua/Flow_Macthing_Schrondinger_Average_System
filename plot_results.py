import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from scipy.stats import gaussian_kde
import ot 

def plot_trajectories(traj, u, u_z, t_grid, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    _,T,control_dim = u.shape
    mse_loss = nn.MSELoss()
    loss_list = []
    for j in range(T):
        loss = mse_loss(u[:,j, :], u_z[:,j, :])
        loss_list.append(loss.detach().cpu().numpy())
        
    plt.figure(figsize=(8,6))
    plt.plot(loss_list)
    plt.savefig(os.path.join(log_dir, 'u_distance.jpg'))
    if torch.is_tensor(traj):
        traj = traj.detach().cpu().numpy()
        u = u.detach().cpu().numpy()
        u_z = u_z.detach().cpu().numpy()
        t_grid = t_grid.detach().cpu().numpy()

    n_samples, _, state_dim = traj.shape
    _,T,control_dim = u.shape
    for dim in range(state_dim):
        plt.figure(figsize=(8,6))
        for sample in range(n_samples):
            plt.plot(t_grid, traj[sample,:, dim])
            plt.xlabel(r'$t$')
            plt.ylabel(fr'$x_{dim}$')
            plt.savefig(os.path.join(log_dir, f'x{dim}.jpg'))
    '''
    for dim in range(control_dim):
        plt.figure(figsize=(8,6))
        for sample in range(n_samples):
            plt.plot(t_grid, u[sample,:, dim], label='real u')
            plt.plot(t_grid, u_z[sample,:, dim], label='predicted u')
            plt.legend()
            plt.xlabel(r'$t$')
            plt.ylabel(fr'$u_{dim}$')
            plt.savefig(log_dir+f'u{dim}.jpg')
    '''

    

def plot_initial_target_dis_2d(x0, xf, GT_traj, traj,t_grid, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    if torch.is_tensor(traj):
        x0 =x0.detach().cpu().numpy()
        xf = xf.detach().cpu().numpy()
        GT_traj = GT_traj.detach().cpu().numpy()
        traj = traj.detach().cpu().numpy()
        t_grid = t_grid.detach().cpu().numpy()

    plt.figure(figsize=(8,6))
    N, nx = x0.shape
    for i in range(N):
        plt.plot(traj[i,:,0], traj[i,:,1], color='m', alpha=0.1)
    x0_pred = traj[:,0,:]
    xf_pred = traj[:,-1,:]
    x0_GT = GT_traj[:,0,:]
    xf_GT = GT_traj[:, -1, :]

    #visualize_2d(x0_pred)
    visualize_2d(xf_pred, 'xf with pred u')
    #visualize_2d(x0_GT)
    visualize_2d(xf_GT, 'xf with real u')
    plt.scatter(x0[:, 0], x0[:, 1], color='b', label='initial distribution')
    plt.scatter(xf[:,0],  xf[:,1], color='r', label='target distribution')
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.legend(loc=2)
    plt.xlabel(r'$x_t(1)$')
    plt.ylabel(r'$x_t(2)$')
    plt.savefig(os.path.join(log_dir, 'traj.jpg'))

    # plot heatmap
    plt.figure(figsize=(8,6))
    plt.hist2d(x0_pred[:,0], x0_pred[:,1], bins=50, density=True, cmap='Blues', label='x0 pred')
    plt.hist2d(xf_pred[:,0], xf_pred[:,1], bins=50, density=True, cmap='Reds', label='xf with pred u')
    #plt.hist2d(x0_GT[:,0], x0_GT[:,1], bins=50, density=True, cmap='Greens', label='x0 GT')
    plt.hist2d(xf_GT[:,0], xf_GT[:,1], bins=50, density=True, cmap='Grays', label='xf with real u')
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.title("2D Histogram density")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig(os.path.join(log_dir,'heatmap.jpg'))

def plot_initial_target_dis_1d(x0, xf, GT_traj, traj, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    if torch.is_tensor(traj):
        x0 =x0.detach().cpu().numpy()
        xf = xf.detach().cpu().numpy()
        traj = traj.detach().cpu().numpy()
        GT_traj = GT_traj.detach().cpu().numpy()
    N, _ = x0.shape
    plt.figure(figsize=(8,6))
    for i in range(N):
        plt.plot(traj[i,:,0], np.zeros_like(traj[i,:]), color='m', alpha=0.1)
    plt.plot(x0, np.zeros_like(x0), 'bo', markersize=2, label='initial')
    plt.plot(xf, np.zeros_like(xf), 'ro', markersize=2, label='target')
    x0_pred = traj[:,0,0]
    xf_pred = traj[:,-1,0]
    #GT_x0 = GT_traj[:,0,0]
    GT_xf = GT_traj[:,-1,0]
    kde0 = gaussian_kde(x0_pred)
    kdef = gaussian_kde(xf_pred)
    plt.hist(x0_pred, bins=30, density=True, alpha=0.4, label='initial distribution')
    plt.hist(xf_pred, bins=30, density=True, alpha=0.4, label='target with pred u')
    #plt.hist(GT_x0, bins=30, density=True, alpha=0.4, label='initial with real u')
    plt.hist(GT_xf, bins=30, density=True, alpha=0.4, label='target with real u')
    #plt.hist(x0, bins=30, density=True, alpha=0.4, label='GT initial')
    plt.hist(xf, bins=30, density=True, alpha=0.4, label='target distribution')
    xs = np.linspace(x0_pred.min(), x0_pred.max(), 200)
    plt.plot(xs, kde0(xs), label='initial KDE')
    xs = np.linspace(xf_pred.min(), xf_pred.max(), 200)
    plt.plot(xs, kdef(xs), label='target KDE')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.legend(loc=2)
    plt.savefig(os.path.join(log_dir,'Density.jpg'))

def plot_loss(loss, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    if torch.is_tensor(loss):
        loss = loss.detach().cpu().numpy()
    epoch, loss = zip(*loss)
    plt.figure(figsize=(8,6))
    plt.plot(list(epoch), list(loss))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(log_dir,'loss.jpg'))

def visualize_2d(pts, label):
    x, y = pts[:, 0], pts[:, 1]
    plt.scatter(x, y, s=10, alpha=0.4, label=f'samples of {label}')
    kde = gaussian_kde(pts.T)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100),
                         np.linspace(ymin, ymax, 100))
    grid = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(grid).reshape(xx.shape)
    plt.contour(xx, yy, zz, levels=6, colors='k', linewidths=1)

'''
W2 distance
'''
def W2_2d(x0, xf, GT_traj, traj,t_grid, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    n_samples, _, state_dim = traj.shape
    if torch.is_tensor(traj):
        x0 =x0.detach().cpu().numpy()
        xf = xf.detach().cpu().numpy()
        GT_traj = GT_traj.detach().cpu().numpy()
        traj = traj.detach().cpu().numpy()
        t_grid = t_grid.detach().cpu().numpy()
    
    xf_pred = traj[:,-1, :]
    w2_dist = []
    a = np.ones(n_samples)/n_samples 
    b = np.ones(n_samples)/n_samples
    '''
    dist_0 =  ot.dist(x0, xf_pred) 
    w2_dist_norm = np.sqrt(ot.emd2(a, b, dist_0))
    
    T = len(t_grid)
    for i in range(len(t_grid)):
        if i% 10 ==0:
            print(f'{i}/{T}')
        dist =  ot.dist(x0[:,i,], xf_pred[:,i,]) 
        w2_dist.append(np.sqrt(ot.emd2(a, b, dist))/w2_dist_norm)

    plt.figure(figsize=(8,6 ))
    plt.plot(t_grid, w2_dist,lw=2)
    plt.xlabel(r'time')
    plt.ylabel(r'$W_2$ distance')
    plt.savefig(os.path.join(log_dir, 'W2.jpg'))
    '''
    M = ot.dist(xf, xf_pred, metric='euclidean')**2
    w2_sq = ot.emd2(a, b, M)
    w2 = np.sqrt(w2_sq)
    print(f"2D W2 distance = {w2:.3f}")


def W2_1d(x0, xf, GT_traj, traj,t_grid, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    n_samples, _, state_dim = traj.shape
    if torch.is_tensor(traj):
        x0 =x0.detach().cpu().numpy()
        xf = xf.detach().cpu().numpy()
        GT_traj = GT_traj.detach().cpu().numpy()
        traj = traj.detach().cpu().numpy()
        t_grid = t_grid.detach().cpu().numpy()
    
    xf_pred = traj[:,-1, :]
    a_sorted = np.sort(xf)
    b_sorted = np.sort(xf_pred)
    w2 = np.sqrt(np.mean((a_sorted - b_sorted)**2))
    print(f"1D W2 distance = {w2:.3f}")