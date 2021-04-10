# -*- coding:utf-8 -*-
import torch.nn as nn
from bertSpytorch.models.LayerNorm import LayerNorm
class ResNetSubLayer(nn.Module):
    def __init__(self,size,drop_out):
        super(ResNetSubLayer,self).__init__()
        self.norm=LayerNorm(size)
        self.dropout=nn.Dropout(drop_out)
    def forward(self, input,sublayer):
        out=self.norm(input+self.dropout(sublayer))
        return out