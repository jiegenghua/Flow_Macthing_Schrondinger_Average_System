'''
example 2
'''
import torch
from torch.distributions import MultivariateNormal

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
    
sys_name = 'Example2'
nx = 2  # dimension of the state
nu = 2 # dimension of the control
A_fn = lambda theta: torch.tensor([[-theta, 0.0], [0.0, -theta]], device=device)
B_fn = lambda theta: torch.eye(nx, device=device)
