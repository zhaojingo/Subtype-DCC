import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import itertools
import numpy as np
import os




def block(in_c,out_c):
    layers=[
        nn.Linear(in_c,out_c),
        nn.ReLU(True)
    ]
    return layers

class Encoder(nn.Module):
    def __init__(self,input_dim=9844,inter_dims=[5000,2000,1000,256]):
        super(Encoder,self).__init__()

        self.encoder=nn.Sequential(
            nn.Dropout(),
            *block(input_dim,inter_dims[0]),
            *block(inter_dims[0],inter_dims[1]),
            *block(inter_dims[1],inter_dims[2]),
            *block(inter_dims[2],inter_dims[3])  
        )


    def forward(self, x):
        z=self.encoder(x)

        return z


class Decoder(nn.Module):
    def __init__(self,input_dim=9844,inter_dims=[5000,2000,1000,256]):
        super(Decoder,self).__init__()

        self.decoder=nn.Sequential(
            *block(inter_dims[-1],inter_dims[-2]),
            *block(inter_dims[-2],inter_dims[-3]),
            *block(inter_dims[-3],inter_dims[-4]) 
        )

    def forward(self, z):
        x_out=self.decoder(z)

        return x_out


class AE(nn.Module):
    def __init__(self,hid_dim=256):
        super(AE,self).__init__()
        self.encoder=Encoder()
        self.decoder=Decoder()

        self.rep_dim = hid_dim

    def forward(self, x):
        z = self.encoder(x)
        x_out = self.decoder(z)
        
        return z



    