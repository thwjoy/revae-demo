import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import os


class REVAE(nn.Module):
    """
    Class that deals with the proposed M3 model
    """
    def __init__(self, **kwargs):
        super(REVAE, self).__init__()
        self.num_classes = 18
        self.z_dim = 45
        self.z_classify = self.num_classes
        self.z_style = self.z_dim - self.z_classify
        self.im_shape = (3, 64, 64)
        self._z_prior_fn = torch.distributions.Normal
        load_data = './data/revae'
        
        self.encoder_z = torch.load(os.path.join(load_data,'encoder_z.pt'))
        self.decoder = torch.load(os.path.join(load_data,'decoder.pt'))
        self.regressor = torch.load(os.path.join(load_data, 'regressor.pt'))


        self.lims = []
        for i in range(self.num_classes):
            mult = 4
            y_1 = torch.zeros(1, self.num_classes)
            locs_false, scales_false, _ = self.regressor(y_1)
            y_1[:, i].fill_(1.0)
            locs_true, scales_true, _ = self.regressor(y_1)
            sign = torch.sign(locs_true[:, i] - locs_false[:, i])
            z_1_false_lim = (locs_false[:, i] + -mult * sign * scales_false[:, i]).item()    
            z_1_true_lim = (locs_true[:, i] + mult * sign * scales_true[:, i]).item()
            self.lims.append([z_1_false_lim, z_1_true_lim])

            

    def _clip_z(self, z):
        return z[:, :self.z_classify]

    def reconstruct_img(self, x):
        loc, scale = self.encoder_z(x)
        z = self._z_prior_fn(loc, scale).sample()
        return self.decoder(z)
