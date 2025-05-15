'''
example 2
'''
import torch
from torch.distributions import MultivariateNormal


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys_name = 'Example2'
nx = 2  # dimension of the state
nu = 2 # dimension of the control
A_fn = lambda theta: torch.tensor([[-theta, 0.0], [0.0, -theta]], device=device)
B_fn = lambda theta: torch.eye(nx, device=device)
