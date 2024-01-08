import torch
import torch.nn as nn
from NSDE_BMM_main.nsde_bmm.architecture import NSDE


class LinearDiffusion(nn.Module):
    def __init__(self, image_h, device="cpu", num_layers=1):
        super().__init__()
        self.device = device
        self.image_h = image_h
        self.len = image_h**2
        self.t_emb = self.time_embeddings()
        self.l1 = nn.Linear(in_features=self.len, out_features=self.len*4, device=device)
        self.act = torch.relu
        self.l2 = nn.Linear(in_features=self.len*4, out_features=self.len*2, device=device)
        self.l3 = nn.Linear(in_features=self.len*2, out_features=self.len, device=device)

    def time_embeddings(self):
        n=10000.0
        T = 1000
        d = self.len #d_model=head_num*d_k, not d_q, d_k, d_v

        positions = torch.arange(0, T).unsqueeze_(1)
        embeddings = torch.zeros(T, d)

        denominators = torch.pow(n, 2*torch.arange(0, d//2)/d) # 10000^(2i/d_model), i is the index of embedding
        embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
        embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))

        return embeddings.to(self.device)

    def forward(self, x, t):
        x_emb = (x.view(-1,self.len) + self.t_emb[t]).view(-1,self.len)
        y = self.act(self.l1(x_emb))
        y= self.act(self.l2(y))
        y = self.l3(y)
        return y.view(-1,1,self.image_h, self.image_h)
    

class DiffusionNSDE(NSDE):
    def __init__(self, d=2, dt=.05, n_hidden=50, p=0.8, device = "cpu"):
        super().__init__(d, dt, n_hidden, p)
        self.device = device
        self.image_h = int(d**0.5)
        self.len = d
        self.t_emb = self.time_embeddings()
        self.dt = self.dt.to(device)

    def time_embeddings(self):
        n=10000.0
        T = 1000
        d = self.d #d_model=head_num*d_k, not d_q, d_k, d_v

        positions = torch.arange(0, T).unsqueeze_(1)
        embeddings = torch.zeros(T, d)

        denominators = torch.pow(n, 2*torch.arange(0, d//2)/d) # 10000^(2i/d_model), i is the index of embedding
        embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
        embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))

        return embeddings.to(self.device)
    
    def next_moments(self, m, P):
        m = m.view(-1,self.len)
        P = P.view(-1,self.len,self.len)
        f_m, P_ff = self.drift_moments(m, P)
        f_m = f_m*self.dt
        P_ff = P_ff*(self.dt**2) # Cov(f)                           
        L_P_central = self.diffusion_central_moment(m, P)*self.dt # E[LL^T]

        Fx = self.exp_jac # Expected Jacobian
        P_xf = Fx@P*(self.dt) # Cov(x,f), Eq. 13
        
        P_nxt = P + P_ff + P_xf + P_xf.transpose(1,2) + L_P_central # Eq. 20
        m_nxt = m + f_m
        return m_nxt.view(-1,1,self.image_h, self.image_h), P_nxt