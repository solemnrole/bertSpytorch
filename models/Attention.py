# -*- coding:utf-8 -*-
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
class MultiHeadAttention(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_attention_heads=1,
                 attention_probs_dropout_prob=0.0,
                 ):
        super(MultiHeadAttention,self).__init__()
        assert hidden_size%num_attention_heads==0
        self.num_attention_heads=num_attention_heads
        self.size_per_head=hidden_size//num_attention_heads
        self.q_kernel=nn.Linear(hidden_size, hidden_size)
        self.k_kernel = nn.Linear(hidden_size, hidden_size)
        self.v_kernel = nn.Linear(hidden_size, hidden_size)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=attention_probs_dropout_prob)
        self.output_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, query,key,value,mask=None):
        seq_length=query.size(1)
        qw=self.q_kernel(query)
        kw=self.k_kernel(key)
        vw=self.v_kernel(value)
        qw = qw.reshape((-1, seq_length, self.num_attention_heads, self.size_per_head))
        kw = kw.reshape((-1, seq_length, self.num_attention_heads, self.size_per_head))
        vw = vw.reshape((-1, seq_length, self.num_attention_heads, self.size_per_head))
        qw=qw.permute(0,2,1,3)
        kw=kw.permute(0,2,3,1)
        vw=vw.permute(0,2,1,3)

        context_layer=self.attention(qw,kw,vw,mask)
        context_layer = context_layer.permute(0, 2, 1, 3)
        context_layer = context_layer.reshape(-1, seq_length, self.num_attention_heads * self.size_per_head)
        return self.output_linear(context_layer)

class Attention(nn.Module):
    def __init__(self,):
        super(Attention,self).__init__()
    def forward(self, query,key,value,mask,dropout=None):
        input_size=query.size()
        scores=torch.mul(torch.matmul(query,key),1./math.sqrt(float(query.size()[-1])))
        if mask is not None:
            addr=((1-mask)*-1e12).reshape(input_size[0],1,input_size[2],1)
            scores+=addr
        p_attn=F.softmax(scores,dim=-1)
        if dropout is not None:
            p_attn=dropout(p_attn)
        return torch.matmul(p_attn,value)