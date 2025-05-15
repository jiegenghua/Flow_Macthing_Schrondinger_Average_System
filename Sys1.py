'''
example 1
'''
import torch
from torch.distributions import MultivariateNormal


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys_name = 'Example1'
nx = 1     # dimension of x
nu =1      # dimension of u
A_fn = lambda theta: torch.tensor([[-theta]], device=device)
B_fn = lambda theta: torch.eye(nx, device=device)

mu0 = MultivariateNormal(torch.zeros(nx), torch.eye(nx))  # initial distribution
muf = MultivariateNormal(torch.ones(nx) * 4, torch.eye(nx))  # target distribution