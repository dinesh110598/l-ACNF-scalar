# %%
from l_acnf import *
import numpy as np
# %%
def symmetry_transform(phi):
    f = [(lambda phi: torch.rot90(phi, k, [0,1]))
         for k in range(4)]
    f += [(lambda phi: torch.flip(phi, dims=[r]))
          for r in range(2)]
    n = np.random.randint(6)
    return f[n](phi)

def sample_symmetric(lattice: Lattice, net: l_ACNF, batch=100,
                     device='cpu'):
    
    phi, phi_lp = sample_fn(lattice, net, batch, device)
    S_scalar = lattice.S(phi)
    phi_lp = phi_lp.unsqueeze(1)
    
    for k in range(1,4):
        phi_rot = torch.rot90(phi, k, [1,2])
        phi_lp1 = logprob_fn(lattice, net, phi_rot).unsqueeze(1)
        phi_lp = torch.cat([phi_lp, phi_lp1], dim=1)
        
    for rdim in [1,2]:
        phi_ref = torch.flip(phi, dims=[rdim])
        phi_lp1 = logprob_fn(lattice, net, phi_ref).unsqueeze(1)
        phi_lp = torch.cat([phi_lp, phi_lp1], dim=1)
    
    phi_lp = torch.logsumexp(phi_lp, 1) - torch.tensor(6.).log()
    
    return phi, phi_lp, S_scalar

def greens_function2(phi, device='cpu'):
    batch = phi.shape[0]
    L = phi.shape[1]
    
    G = torch.zeros([batch, L, L], device=device)
    for t in range(L):
        for x in range(L):
            #phi rolled by t and x along their axes
            phi_roll = torch.roll(phi, shifts=(t,x), dims=(1,2))
            G[:, t, x] = (phi*phi_roll).mean([1,2])
    return G

def metropolis(lattice: Lattice, net: l_ACNF, batch=200,
               steps=10000, device='cpu'):
    epochs = steps//batch
    acc = 0
    curr_x = None
    curr_model_lp = None
    curr_target_lp = None

    phi2 = []
    chi_1 = []
    chi_2 = []
    E = []
    rej = []
    
    for ep in range(epochs):
        with torch.no_grad():
            phi, phi_lp, S = sample_symmetric(lattice, net,
                                              batch, device)
            # phi, phi_lp = sample_fn(lattice, net, batch, device)
            # S = lattice.S(phi)
        rejection = torch.zeros([batch])
        
        if ep==0:
            curr_x = phi[0]
            curr_model_lp = phi_lp[0]
            curr_target_lp = -S[0]
            start = 1
        else:
            start = 0

        for i in range(start, batch):
            indx = ep*batch + i
            
            prop_x = phi[i]
            prop_model_lp = phi_lp[i]
            prop_target_lp = -S[i]
            acc_prob = torch.exp(curr_model_lp - prop_model_lp +
                                 prop_target_lp - curr_target_lp)

            if np.random.rand() < acc_prob:
                curr_x = prop_x
                curr_model_lp = prop_model_lp
                curr_target_lp = prop_target_lp
                acc += 1
            else:
                rejection[i] = 1
            phi[i, :, :] = symmetry_transform(curr_x)
        
        phi2.append(phi)
        rej.append(rejection)
        G = greens_function2(phi, device)
        chi_1.append(G[:, 0, 0])
        chi_2.append(G.sum([1,2]))
        E.append((G[:, 0, 1] + G[:, 1, 0])/2.)
    
    phi2 = torch.cat(phi2, dim=0)
    
    rej = torch.cat(rej, dim=0)
    chi_1 = torch.cat(chi_1, dim=0)
    chi_2 = torch.cat(chi_2, dim=0)
    E = torch.cat(E, dim=0)
    
    return phi2, rej, chi_1, chi_2, E

def greens_function(phi, device='cpu'):
    batch = phi.shape[0]
    L = phi.shape[1]
    
    G = torch.zeros([L, L], device=device)
    for t in range(L):
        for x in range(L):
            #phi rolled by t and x along their axes
            phi_roll = torch.roll(phi, shifts=(t,x), dims=(1,2))
            cross = (phi*phi_roll).mean(0)
            prod = phi.mean(0)*phi_roll.mean(0)
            G[t, x] = (cross - prod).mean()
    return G

def pole_mass(G):
    return torch.acosh((G[:-2]+G[2:])/(2*G[1:-1]))

def susceptibilty(G):
    return torch.sum(G)

def ising_energy(G):
    return (G[0,1] + G[1,0])/2
# %%
# device = 'cuda:0'
# net = l_ACNF(12, 128).to(device)
# net.load_state_dict(torch.load("lacnf_d_8_L20.pth"))
# # %%
# m_p_mean = torch.zeros([7])
# m_p_std = torch.ones([7])

# chi_2_mean = torch.zeros([7])
# chi_2_std = torch.ones([7])

# E_mean = torch.zeros([7])
# E_std = torch.ones([7])

# rej = []
# chi_1 = []
# chi_2 = []
# E = []
# # %%
# g = [6.008, 5.55, 5.276, 5.113, 4.99, 4.89, 4.82]
# T = 100_000

# for (j,L) in enumerate(range(8, 21, 2)):
#     lattice = Lattice(L, g[j], True)
#     phi, r, c1, c2, e = metropolis(lattice, net, 2500, T, device)
    
#     rej.append(r)
#     chi_1.append(c1)
#     chi_2.append(c2)
#     E.append(e)
    
#     # Moving block bootstrap
#     bin = 100
#     boxes = [torch.arange(i, i+bin, device=device) 
#             for i in range(T-bin+1)]
    
#     m_p = []
#     chi_22 = torch.zeros([bin])
#     E2 = torch.zeros([bin])
    
#     for n in range(bin):
#         r = np.random.randint(0, T-bin+1, (T//bin,))
#         indx = torch.cat([boxes[n] for n in r], dim=0)
        
#         phi2 = phi[indx, :, :]
#         G = greens_function(phi2, device)
#         G2 = G.mean(1)
        
#         m_p.append(pole_mass(G2))
#         chi_22[n] = susceptibilty(G)
#         E2[n] = ising_energy(G)
    
#     m_p = torch.stack(m_p, dim=0)
#     m_p_mean[j] = m_p.mean(0)[1:].mean()*L
#     m_p_std[j] = m_p.std(0)[1:].mean()*L
    
#     chi_2_mean[j] = chi_22.mean()
#     chi_2_std[j] = chi_22.std()
#     E_mean[j] = E2.mean()
#     E_std[j] = E2.std()

# rej = torch.stack(rej, dim=0)
# chi_1 = torch.stack(chi_1, dim=0)
# chi_2 = torch.stack(chi_2, dim=0)
# E = torch.stack(E, dim=0)
# # %%
# t_max = 100
# rho_acc = torch.zeros([rej.shape[0], t_max])
# for t in range(1, t_max+1):
#     pool_op = torch.nn.MaxPool1d(t, 1)
#     rho_acc[:, t-1] = (1-pool_op((1-rej).unsqueeze(1))).mean([1,2])
# t_int_acc = 0.5 + rho_acc.sum(1)
# # %%
# t_max = 100
# rho_chi_1 = torch.zeros([chi_1.shape[0], t_max])
# rho_chi_2 = torch.zeros([chi_2.shape[0], t_max])
# rho_E = torch.zeros([E.shape[0], t_max])

# for t in range(1, t_max+1):
#     chi_1_mean2 = chi_1.mean(1, keepdim=True)
#     covar = (chi_1[:, :-t]-chi_1_mean2)*(chi_1[:, t:]-chi_1_mean2)
#     rho_chi_1[:, t-1] = covar.mean(1)/chi_1.var(1)
    
#     chi_2_mean2 = chi_2.mean(1, keepdim=True)
#     covar = (chi_2[:, :-t]-chi_2_mean2)*(chi_2[:, t:]-chi_2_mean2)
#     rho_chi_2[:, t-1] = covar.mean(1)/chi_2.var(1)
    
#     E_mean2 = E.mean(1, keepdim=True)
#     covar = (E[:, :-t]-E_mean2)*(E[:, t:]-E_mean2)
#     rho_E[:, t-1] = covar.mean(1)/E.var(1)

# t_int_chi_1 = 0.5 + rho_chi_1.sum(1)
# t_int_chi_2 = 0.5 + rho_chi_2.sum(1)
# t_int_E = 0.5 + rho_E.sum(1)
# # %%
