from braindecode.models.shallow_fbcsp import ShallowFBCSPNet

import torch.nn as nn

from torch import nn
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Sequential):
    def __init__(self, embedding_size=40):
        super().__init__()
        
        shallownet = ShallowFBCSPNet(22, 40, conv_nonlin=nn.ELU(), pool_nonlin=nn.Identity(), first_conv_bias=True)

        new_shallownet = nn.Sequential()
        for name, module_ in shallownet.named_children():
            if "conv_classifier" in name:
                break
            new_shallownet.add_module(name, module_)

        self.add_module("ShallowNet", new_shallownet)
        self.add_module("Projection", nn.Sequential(
            nn.Conv2d(40, embedding_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        ))
