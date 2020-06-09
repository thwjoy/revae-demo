import torch
import torch.nn as nn
import torch.nn.functional as F
            
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
        
class CELEBAEncoder(nn.Module):
    def __init__(self, z_dim, hidden_dim=256, *args, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),    
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),   
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),         
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),       
            nn.ReLU(True),
            nn.Conv2d(128, hidden_dim, 4, 1),            
            nn.ReLU(True),
            View((-1, hidden_dim*1*1)),               
            nn.Linear(hidden_dim, z_dim*2),            
        )

        self.locs = nn.Linear(hidden_dim, z_dim)
        self.scales = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        hidden = self.encoder(x)
        return hidden[:, :self.z_dim], torch.exp(hidden[:, self.z_dim:])

class CELEBADecoder(nn.Module):
    def __init__(self, z_dim, hidden_dim=256, *args, **kwargs):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),     
            View((-1, hidden_dim, 1, 1)),              
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, 128, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), 
            nn.Sigmoid()
        )

    def forward(self, z):
        m = self.decoder(z)
        return m

class Indep_Regressor(nn.Module):
    def __init__(self, dim):
        super(Indep_Regressor, self).__init__()
        self.dim = dim
        self.diag_loc_true = nn.Parameter(torch.zeros(self.dim))
        self.diag_loc_false = nn.Parameter(torch.zeros(self.dim))
        self.diag_scale_true = nn.Parameter(torch.ones(self.dim))
        self.diag_scale_false = nn.Parameter(torch.ones(self.dim))

    def forward(self, x):
        loc = x * self.diag_loc_true + (1 - x) * self.diag_loc_false
        scale = x * self.diag_scale_true + (1 - x) * self.diag_scale_false
        return loc, torch.clamp(F.softplus(scale), min=1e-3), None
