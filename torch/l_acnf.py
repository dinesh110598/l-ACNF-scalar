# %%
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as dist
import numpy as np
# %%
class Lattice:
    "Parameters for 2D lattice field and accompanying methods"
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
    
    def ESS(self, phi, phi_lp):
        KL = self.KL_loss(phi, phi_lp)
        arg = 2*torch.logsumexp(-KL, 0) - torch.logsumexp(-2*KL, 0)
        return torch.exp(arg)/phi.shape[0]
    
    def pos_t_param(self, batch, device='cpu'):
        L = self.L
        
        pos_t = torch.arange(L, device=device).float().unsqueeze(1)/L
        pos_t = pos_t.expand(batch, L, L)
        pos_flag = torch.zeros((batch, L, L), device=device)
        pos_flag[:, 0, :] = 1
        
        param = torch.full((batch, 1, L), self.g, device=device)
        
        return pos_t, pos_flag, param
    
def concat_elu(x):
    return F.elu(torch.cat([x, -x], dim=1))

def generate_mask(L):
    return torch.arange(L)%2

class GatedConv2(nn.Module):
    "Gated convolution layer with skip connections for conditional inputs"
    def __init__(self, in_channels, out_channels, width=64,
                 s=True):
        super(GatedConv2, self).__init__()
        
        self.s = s
        self.conv = nn.Conv1d(1, width//2, 3,
                              padding='same', padding_mode='circular')
        # self.bn = nn.BatchNorm1d(in_channels//2)
        self.conv1 = nn.Conv1d(width, width, 1)
        self.relu = nn.LeakyReLU()
        
        self.skip_net = nn.Conv1d(width, width//2, 1)
        self.conv2 = nn.Conv1d(width, 2*width, 3,
                               padding='same', padding_mode='circular')
        self.conv3 = nn.Conv1d(width, out_channels, 1)
        
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

class GatedConv(nn.Module):
    "Gated convolution layer with optional skip connections for residual inputs"
    def __init__(self, in_channels, out_channels,
                 skip=False):
        super(GatedConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, in_channels//2, 3, 
                              padding='same', padding_mode='circular')
        
        self.bn = nn.BatchNorm1d(in_channels//2)
        self.conv1 = nn.Conv1d(in_channels, in_channels, 1)
        self.relu = nn.LeakyReLU()
        if skip:
            self.skip_net = nn.Conv1d(in_channels, in_channels//2, 1)
        self.conv2 = nn.Conv1d(in_channels, 2*out_channels, 1)
        
    def forward(self, x, a=None, last=False):
        y = self.conv(x)
        y = self.bn(y)
        if a is not None:
            y += self.skip_net(a)
        y = concat_elu(y)
        
        y = self.relu(self.conv1(y))
        
        y = self.conv2(y)
        y1, y2 = torch.chunk(y, 2, dim=1)
        y = y1 * torch.sigmoid(y2)
        if last:
            return y
        else:
            return y + x
        

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
        m1 = (torch.arange(L, device=x.device)%2).expand(1, 1, L)
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
            s_vals.append(s.sum(2))
        return x, torch.cat(s_vals, 1).sum(1)
    
    def inv(self, x, theta):
        L = x.shape[-1]
        m1 = (torch.arange(L, device=x.device)%2).expand(1, 1, L)
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
            s_vals.append(s.sum(2))
        return x, torch.cat(s_vals, 1).sum(1)
    
class l_ACNF(nn.Module):
    def __init__(self, depth, width):
        super().__init__()        
        init_layer = [nn.Conv1d(5, width, 3, padding='same',
                                padding_mode='circular'),
                      nn.ELU()]
        nets = nn.ModuleList([nn.Sequential(*init_layer)])
        nets.append(GatedConv(width, width))
        
        nets.append(GatedConv(width, width))
        
        nets.append(GatedConv(width, width, skip=True))    
        nets.append(GatedConv(width, width, skip=True))
        
        self.nets = nets
        self.realnvp = RealNVP(depth, width)
        
    def forward(self, x):
        # x has previous phi, pos_t and g concatenated
        batch = x.shape[0]
        L = x.shape[2]
        
        stream = [self.nets[0](x)]
        stream += [self.nets[1](stream[-1])]
        
        stream += [self.nets[2](stream[-1])]
        
        stream += [self.nets[3](stream[-1], stream[1])]
        stream += [self.nets[4](stream[-1], stream[0], True)]
        x2 = stream[-1]
        
        mu = torch.zeros([batch, L], device=x.device)
        sig = torch.ones_like(mu)
        gauss = dist.Independent(dist.Normal(mu, sig), 1)
        gauss_x = gauss.sample()
        gauss_lp = gauss.log_prob(gauss_x)
        
        z, logdetJ = self.realnvp(gauss_x.unsqueeze(1), x2)
        
        logp_z = (gauss_lp - logdetJ)
        return z, logp_z

    def logprob(self, z, x):
        batch = x.shape[0]
        L = x.shape[2]
        
        stream = [self.nets[0](x)]
        stream += [self.nets[1](stream[-1])]
        
        stream += [self.nets[2](stream[-1])]
        
        stream += [self.nets[3](stream[-1], stream[1])]
        stream += [self.nets[4](stream[-1], stream[0], True)]
        x = stream[-1]
        
        mu = torch.zeros([batch, L], device=x.device)
        sig = torch.ones_like(mu)
        gauss = dist.Independent(dist.Normal(mu, sig), 1)
        
        gauss_x, logdetJ_inv = self.realnvp.inv(z, x)
        gauss_lp = gauss.log_prob(gauss_x[:, 0, :])
        logp_z = (gauss_lp + logdetJ_inv)
        return logp_z
# %%
def sample_fn(lattice: Lattice, net: l_ACNF, batch=100,
              device='cpu'):
    L = lattice.L
    
    phi = torch.zeros((batch, 0, L), device=device)
    phi_lp = torch.zeros((batch,), device=device)
    pos_t, pos_flag, param = lattice.pos_t_param(batch, device)
    
    for t in range(L):
        with torch.no_grad():
            if t==0:
                dep_set = torch.zeros([batch, 2, L], 
                                      device=device)
            else:
                dep_set = torch.cat([phi[:, t-1:t, :],
                                     phi[:, :1, :]], dim=1)
        dep_t = pos_t[:, t:t+1, :]
        dep_flag = pos_flag[:, t:t+1, :]
        dep_g = param
        dep_set = torch.cat([dep_set, dep_t, 
                             dep_flag, dep_g], dim=1)
        
        sample, logprob = net(dep_set)
        phi = torch.cat([phi, sample], dim=1)
        phi_lp += logprob
        
    return phi, phi_lp

def logprob_fn(lattice: Lattice, net: l_ACNF, phi):
    L = lattice.L
    batch = phi.shape[0]
    device = phi.device
    
    phi_lp = torch.zeros((batch,), device=device)
    pos_t, pos_flag, param = lattice.pos_t_param(batch, device)
    
    for t in range(L):
        with torch.no_grad():
            if t==0:
                dep_set = torch.zeros([batch, 2, L], 
                                      device=device)
            else:
                dep_set = torch.cat([phi[:, t-1:t, :],
                                     phi[:, :1, :]], dim=1)
        dep_t = pos_t[:, t:t+1, :]
        dep_flag = pos_flag[:, t:t+1, :]
        dep_g = param
        dep_set = torch.cat([dep_set, dep_t,
                             dep_flag, dep_g], dim=1)
        
        logprob = net.logprob(phi[:, t:t+1, :], dep_set)
        phi_lp += logprob
    return phi_lp

def train(lattice: Lattice, net: l_ACNF, batch=100,
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
            torch.save(net.state_dict(), "Saves/train_chkpt.pt")
# %%
# lattice = Lattice(16, 5.0, True)
# device = 'cuda:0'
# net = l_ACNF(8, 128).to(device)
# # %%
# epochs = 1
# batch = 128
# train(lattice, net, batch, epochs, 4e-4, 1500, device)
# %%
# with torch.no_grad():
#     phi, phi_lp = sample_fn(lattice, net, 100, device)
#     loss = lattice.KL_loss(phi, phi_lp)
#     print(loss.mean(), loss.std())
# %%
# torch.save(net.state_dict(), "l_acnf_d_8_L20.pth")
# net.load_state_dict(torch.load("l_acnf_d_8_L20.pth"))
# %%
