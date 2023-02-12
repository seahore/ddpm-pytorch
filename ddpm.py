import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPMTrainer(nn.Module):
    def __init__(self, T, beta_1, beta_T, net):
        """
        Create a Gaussian diffusion training module.

        Args:
            `T` (`int`): Total diffusion steps, i.e. T according to the paper.
            `beta_1`, `beta_T` (`float`): Hyperparameters β_1 and β_T. The rest of βs can be inferred from T and them.
            `net` (`torch.nn.Module`): The model to learn the Gaussian noise distribution.
        """
        super().__init__()
        self.T = T
        self.net = net
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas_bar = torch.cumprod(1. - self.betas, dim=0)
        # c1: $\sqrt{\bar{\alpha}_t}$, the coefficient of x_0 in Algo. 1 Line 5
        # c2: $\sqrt{1-bar{\alpha}_t}$, the coefficient of the latter ε in Algo. 1 Line 5
        self.register_buffer('c1', torch.sqrt(alphas_bar))
        self.register_buffer('c2', torch.sqrt(1. - alphas_bar))
    
    def forward(self, x_0):
        """
        The loop of the Algo. 1 in the paper
        """
        # L2: sampling x_0 from q(x_0), in practice, means selecting an image from the dataset
        # L3: t ~ Uniform({1, ..., T})
        t = torch.randint(self.T, size=(1,), device=x_0.device)
        # L4: ε ~ N(0, I)
        eps = torch.randn_like(x_0)
        # L5: gradient descend on loss
        x_t = self.c1[t] * x_0 + self.c2[t] * eps
        loss = F.mse_loss(eps, self.net(x_t.float(), t))
        return loss
        
class DDPMSampler(nn.Module):
    def __init__(self, T, beta_1, beta_T, net, size):
        self.net = net
        self.size = size
    
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. / self.betas
        self.register_buffer('c1', 1. / torch.sqrt(alphas))
        self.register_buffer('c2', (1. - alphas) / torch.sqrt(1. - torch.cumprod(self.alphas)))
        # just let σ^2 = β_t according to the paper
        self.register_buffer('c3', torch.sqrt(self.betas))
    
    def forward(self):
        """
        Algo. 2 in the paper
        """
        # L1: x_T ~ N(0, I)
        x_T = torch.randn(self.size)
        # L2: for t = T, ..., 1 do
        x_t = x_T
        for t in reversed(range(self.T)):
            # L3: z ~ N(0, I) if t > 1, else z = 0
            z = torch.randn(self.size) if t > 1 else 0
            # L4: x_{t-1} = ...
            x_t = self.c1[t] * (x_t - self.c2[t] * self.net(x_t.float(), t)) + self.c3[t] * z  
        # L5: end for
        # L6: return x_0
        return x_t
