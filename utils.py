import torch
from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily
from sklearn import datasets
import numpy as np

def sample_2G(N, nx, device='cpu'):
    pi = torch.tensor([0.5, 0.5], device=device)
    mu_pos = torch.ones(nx, device=device)*6.0
    mu_neg = -mu_pos
    mus = torch.stack([mu_pos, mu_neg], dim=0)
    convs = torch.eye(nx, device=device).unsqueeze(0).expand(2,nx,nx)
    component_dist = MultivariateNormal(mus, covariance_matrix=convs)
    gmm = MixtureSameFamily(
        mixture_distribution = Categorical(pi),
        component_distribution=component_dist
    )
    return gmm.sample((N,))

def sample_4G(N, nx, device='cpu'):
    pi = torch.full((4,), 0.25, device=device)
    m = torch.ones(nx, device=device)*6.0
    mus = torch.stack([
        +m,
        torch.cat([m[:1],-m[1:]]),
        torch.cat([-m[:1],m[1:]]),
        -m
    ], dim=0)
    convs = torch.eye(nx, device=device).unsqueeze(0).expand(4,nx,nx)
    comp = MultivariateNormal(mus, covariance_matrix=convs)
    gmm = MixtureSameFamily(
        mixture_distribution=Categorical(pi),
        component_distribution=comp
    )
    return gmm.sample((N,))

def Circle(N, R=1.0, device='cpu'):
    theta = 2*torch.pi*torch.rand(N, device=device)
    x = R*torch.cos(theta)+6
    y = R*torch.sin(theta)+6
    return torch.stack([x,y], dim=1)

def HalfMoon(N, device='cpu'):
    noise = 0.1  # noise to add to sample locations
    x, _ = datasets.make_moons(n_samples=N, noise=noise)
    return torch.tensor(x, dtype=torch.float32, device=device)