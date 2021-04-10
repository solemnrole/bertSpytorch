import torch.nn as nn
import torch

class LayerNorm(nn.Module):
    def __init__(self,size,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.weight=nn.Parameter(torch.ones(size))
        self.bias=nn.Parameter(torch.zeros(size))
        self.eps=eps
    def forward(self,input):
        mean=torch.mean(input,dim=-1,keepdim=True)
        std=torch.std(input,dim=-1,keepdim=True)
        return self.weight*(input-mean)/(std+self.eps)+self.bias
