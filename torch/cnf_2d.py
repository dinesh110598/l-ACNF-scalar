# %%
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as dist
import numpy as np
# %%
class Lattice:
    def __init__(self, L, g, pbc=True , m_2=-4):
        self.L = L
        self.m_2 = m_2
        self.g = g
        self.pbc = pbc
        
    def S(self, phi):
        m_2 = self.m_2
        g = self.g
        
        phi_2 = m_2*torch.sum(phi**2, dim=(1,2))
        phi_4 = g*torch.sum(phi**4, dim=(1,2))
        
        if self.pbc:
            phi_t_p = torch.roll(phi, 1, 1)
            phi_t_n = torch.roll(phi, -1, 1)
            d_phi = torch.sum(phi*(2*phi - phi_t_p - phi_t_n),
                              dim=(1,2))
            
            phi_x_p = torch.roll(phi, 1, 2)
            phi_x_n = torch.roll(phi, -1, 2)
            d_phi += torch.sum(phi*(2*phi - phi_x_p - phi_x_n),
                               dim=(1,2))
        
        else:        
            d_phi = torch.sum(phi[:,1:-1,:]*(2*phi[:,1:-1,:] - 
                            phi[:,:-2,:] - phi[:,2:,:]), dim=(1,2))
            d_phi += torch.sum(phi[:,:,1:-1]*(2*phi[:,:,1:-1] -
                            phi[:,:,:-2] - phi[:,:,2:]), dim=(1,2))
        
        return d_phi + phi_2 + phi_4
    
    def KL_loss(self, phi, phi_lp):
        S = self.S(phi)
        return phi_lp + S

def concat_elu(x):
    return F.elu(torch.cat([x, -x], dim=1))

def generate_mask(L, dev):
    mesh = torch.meshgrid(torch.arange(L, device=dev), 
                          torch.arange(L, device=dev), indexing='xy')
    return torch.stack(mesh).sum(0)%2

class GatedConv2(nn.Module):
    def __init__(self, in_channels, out_channels, width=64,
                 s=True):
        super(GatedConv2, self).__init__()
        
        self.s = s
        self.conv = nn.Conv2d(1, width//2, 3,
                              padding='same', padding_mode='circular')
        # self.bn = nn.BatchNorm1d(in_channels//2)
        self.conv1 = nn.Conv2d(width, width, 1)
        self.relu = nn.LeakyReLU()
        
        self.skip_net = nn.Conv2d(width, width//2, 1)
        self.conv2 = nn.Conv2d(width, 2*width, 3,
                               padding='same', padding_mode='circular')
        self.conv3 = nn.Conv2d(width, out_channels, 1)
        
    def forward(self, x, a=None):
        y = self.conv(x)
        # y = self.bn(y)
        if a is not None:
            y += self.skip_net(a)
        y = concat_elu(y)
        
        y = self.relu(self.conv1(y))
        
        y = self.conv2(y)
        a, b = torch.chunk(y, 2, dim=1)
        y = a * torch.sigmoid(b)
        
        y = self.conv3(y)
        
        if self.s:
            y = torch.tanh(y)
        return y
    
class RealNVP(nn.Module):
    def __init__(self, ncouplings=4, width=64):
        super().__init__()
        self.ncouplings = ncouplings
        self.s = nn.ModuleList([GatedConv2(1, 1, width, True) 
                                for _ in range(ncouplings)])
        self.t = nn.ModuleList([GatedConv2(1, 1, width, False) 
                                for _ in range(ncouplings)])
        
        self.s_scale = nn.ParameterList([
            nn.Parameter(torch.randn([])) 
            for _ in range(ncouplings)])
    
    def forward(self, x, theta):
        L = x.shape[-1]
        m1 = generate_mask(L, x.device).expand(1, 1, L, L)
        m2 = 1 - m1
        
        s_vals = []
        for i in range(self.ncouplings):
            # Masks interchanged between even and odd layers
            if i%2==0:
                s = m2*self.s_scale[i]*self.s[i](m1*x, theta)
                x = m1*x + m2*(x*torch.exp(s) + 
                               self.t[i](m1*x, theta))
            else:
                s = m1*self.s_scale[i]*self.s[i](m2*x, theta)
                x = m2*x + m1*(x*torch.exp(s) + 
                               self.t[i](m2*x, theta))
            # Make sure s is masked before summing here
            s_vals.append(s.sum([2,3]))
        return x, torch.cat(s_vals, 1).sum(1)
    
    def inv(self, x, theta):
        L = x.shape[-1]
        m1 = generate_mask(L, x.device).expand(1, 1, L, L)
        m2 = 1 - m1
        
        s_vals = []
        for i in reversed(range(self.ncouplings)):
            # Masks interchanged between even and odd layers
            if i%2==0:
                s = -m2*self.s_scale[i]*self.s[i](m1*x, theta)
                x = m1*x + m2*(x - self.t[i](m1*x, theta))*s.exp()
            else:
                s = -m1*self.s_scale[i]*self.s[i](m2*x, theta)
                x = m2*x + m1*(x - self.t[i](m2*x, theta))*s.exp()
            # Make sure s is masked before summing here
            s_vals.append(s.sum([2,3]))
        return x, torch.cat(s_vals, 1).sum(1)
    
class CNF_2D(nn.Module):
    def __init__(self, depth, width):
        super().__init__()        
        self.init_layer = nn.Sequential(
            nn.Conv2d(1, width, 1),
            nn.ELU())
        
        self.realnvp = RealNVP(depth, width)
        
    def forward(self, x):
        # x has previous phi, pos_t and g concatenated
        batch = x.shape[0]
        L = x.shape[2]
        x2 = self.init_layer(x)
        
        mu = torch.zeros([batch, L, L], device=x.device)
        sig = torch.ones_like(mu)
        gauss = dist.Independent(dist.Normal(mu, sig), 2)
        gauss_x = gauss.sample()
        gauss_lp = gauss.log_prob(gauss_x)
        
        z, logdetJ = self.realnvp(gauss_x.unsqueeze(1), x2)
        
        logp_z = (gauss_lp - logdetJ)
        return z[:, 0, :, :], logp_z

    def logprob(self, z, x):
        batch = x.shape[0]
        L = x.shape[2]
        
        x = self.init_layer(x)
        
        mu = torch.zeros([batch, L, L], device=x.device)
        sig = torch.ones_like(mu)
        gauss = dist.Independent(dist.Normal(mu, sig), 2)
        
        z = z.unsqueeze(1)
        gauss_x, logdetJ_inv = self.realnvp.inv(z, x)
        gauss_lp = gauss.log_prob(gauss_x[:, 0, :])
        logp_z = (gauss_lp + logdetJ_inv)
        return logp_z
# %%
def sample_fn(lattice: Lattice, net: CNF_2D, batch=100, device='cpu'):
    L = lattice.L
    g = lattice.g
    
    theta = torch.full((batch, 1, L, L), g, device=device)
    
    return net(theta)

def logprob_fn(lattice: Lattice, net: CNF_2D, phi):
    L = lattice.L
    g = lattice.g
    batch = phi.shape[0]
    
    theta = torch.full((batch, 1, L, L), g, device=phi.device)
    return net.logprob(phi, theta)

def train(lattice: Lattice, net: CNF_2D, batch=100,
          epochs=100, lr=0.01, schedule_int=400, device='cpu'):
    optimizer = torch.optim.Adam(net.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_int, 
                                                0.4)
    rng = np.random.default_rng()

    for ep in range(epochs):
        # Randomly sample couplings g
        lattice.g = rng.uniform(4.75, 5.45)
        # Zero your gradients for every batch
        optimizer.zero_grad()
        # Obtain phi and phi_lp
        phi, phi_lp = sample_fn(lattice, net, batch, device)
        # Compute loss and gradients
        l = lattice.KL_loss(phi, phi_lp).mean()
        l.backward()
        # Adjust network parameters using optimizer and gradients
        optimizer.step()
        scheduler.step()
    
        if (ep+1) % 200 == 0:
            loss = lattice.KL_loss(phi, phi_lp)
            print('loss_mean: {}'.format(loss.mean()))
            print('loss_std: {}'.format(loss.std()))
        if (ep+1)%800 == 0:
            torch.save(net.state_dict(), "Saves/chkpt_d_8.pt")
# %%
# lattice = Lattice(16, 5.0, True)
# device = 'cuda:0'
# net = CNF_2D(12, 128).to(device)
# # %%
# epochs = 4000
# batch = 128
# train(lattice, net, batch, epochs, 5e-5, 2000, device)
# %%
# torch.save(net.state_dict(), "d_12_2d.pth")
# net.load_state_dict(torch.load("d_12_2d.pth"))
# %%
