'''
example 1
'''
import torch
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
sys = 'Example1'
nx = 1 # dimension of x
nu =1  # dimension of u
A_fn = lambda theta: torch.tensor([[-theta]], device=device)
B_fn = lambda theta: torch.eye(nx, device=device)