#-*- coding:utf-8 -*-
import torch.nn as nn
import torch
from bertSpytorch.models.Transformer import Transformer
from bertSpytorch.models.LayerNorm import LayerNorm
from bertSpytorch.models.Attention import MultiHeadAttention
from bertSpytorch.models.ResNetSubLayer import ResNetSubLayer


class BertEmbedding(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 max_position_embeddings,
                 token_type_vocab_size=2,
                 initializer_range=0.02,
                 dropout=0.1,
                 ):
        super(BertEmbedding,self).__init__()
        '''初始化词向量分布'''
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size,
                                            _weight=
                                            self.truncated_normal_(
                                                torch.zeros(vocab_size, hidden_size),
                                                std=initializer_range)
                                            )
        '''初始化token_type分布分布'''
        self.token_type_embeddings = nn.Embedding(token_type_vocab_size,
                                                  hidden_size,
                                                  _weight=
                                                  self.truncated_normal_(
                                                      torch.zeros(token_type_vocab_size, hidden_size),
                                                      std=initializer_range)
                                                  )

        '''初始化位置向量分布'''
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size,
                                                _weight=
                                                self.truncated_normal_(
                                                    torch.zeros(max_position_embeddings,hidden_size),
                                                    std=initializer_range)
                                                )
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def truncated_normal_(self, tensor, mean=0, std=0.09):
        '''模拟截断的正态分布，截断的正态分布意思是后千分之三是小概率事件，我们认为不发生，将末尾的千分之三砍掉！
        正态分布的数学期望值或期望值 μ 等于位置参数，决定了分布的位置；其方差\sigma^2的开平方或标准差 σ 等于尺度参数，决定了分布的幅度。

        常考虑一组数据具有近似于正态分布的概率分布。
        若其假设正确，则
        约68.3%数值分布在距离平均值有1个标准差 σ 之内的范围
        约95.4%数值分布在距离平均值有2个标准差 σ 之内的范围
        约99.7%数值分布在距离平均值有3个标准差 σ 之内的范围
        剩下千分之3称之为小概率事件，实际使用通常忽略不计。
        '''
        with torch.no_grad():
            size = tensor.size()

            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def apply_embeddings(self, inputs):
        '''bert输入为token_ids,segment_ids'''
        token_ids, segment_ids = inputs
        input_shape = token_ids.size()
        seq_length = input_shape[1]
        '''词向量'''
        word_embedding_out = self.word_embeddings(token_ids)
        '''token_type向量'''
        segment_embedding_out = self.token_type_embeddings(segment_ids)
        '''位置向量'''
        positions = torch.range(0, seq_length - 1, dtype=torch.int64).unsqueeze(dim=0)
        position_embedding_out = self.position_embeddings(positions)
        embedding_out = torch.add(word_embedding_out, segment_embedding_out)
        embedding_out = torch.add(embedding_out, position_embedding_out)

        return self.dropout(embedding_out)

    def forward(self, inputs_ids, token_type_ids):
        embedding_out = self.apply_embeddings((inputs_ids, token_type_ids))
        return embedding_out,self.word_embeddings.weight

class FeedForward(nn.Module):
    def __init__(self,hidden_size,intermediate_size,dropout=0.1):
        super(FeedForward,self).__init__()
        self.ffnc=nn.Sequential(
            nn.Linear(hidden_size,intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size,hidden_size),
        )
    def forward(self, attention_output):
        ffn_out=self.ffnc(attention_output)
        return ffn_out

class EncoderBlock(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_probs_dropout_prob,
                 hidden_dropout_prob,
                 intermediate_size,
                 ):
        super(EncoderBlock,self).__init__()
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )
        self.attention_resNetSubLayer_input = ResNetSubLayer(
            size=hidden_size,
            drop_out=hidden_dropout_prob
        )
        self.feedForward = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=hidden_dropout_prob
        )
        self.ffn_resNetSubLayer_attention = ResNetSubLayer(
            size=hidden_size,
            drop_out=hidden_dropout_prob
        )

    def forward(self, input_tensor, mask=None):
        multiAttn_out = self.attention(input_tensor,input_tensor,input_tensor,mask)
        attention_resNetSubLayer_input=self.attention_resNetSubLayer_input(
            input_tensor,
            multiAttn_out
        )
        feedForward=self.feedForward.forward(attention_resNetSubLayer_input)
        ffn_resNetSubLayer_attention=self.ffn_resNetSubLayer_attention(
            attention_resNetSubLayer_input,
            feedForward)
        return ffn_resNetSubLayer_attention

class MLM(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 ):
        super(MLM,self).__init__()
        self.hidden_size=hidden_size
        self.vocab_size=vocab_size
        self.mlm_linear=nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.GELU(),
            LayerNorm(hidden_size)
        )
        self.mlm_output_bias=nn.Parameter(torch.zeros(vocab_size))
        self.mlm_logsoftmax=nn.LogSoftmax(dim=-1)

    def forward(self, input_tensor,word_embeddings_table):
        """output_weights共用wordEmbedding"""
        input_shape=input_tensor.size()
        input_tensor=self.mlm_linear(input_tensor)
        output_tensor=torch.matmul(input_tensor.reshape(-1,self.hidden_size),word_embeddings_table.permute(1,0))
        output_tensor=torch.add(output_tensor,self.mlm_output_bias)
        log_probs=self.mlm_logsoftmax(output_tensor).reshape(input_shape[0],input_shape[1],self.vocab_size)
        # log_probs=output_tensor.reshape([input_shape[0],input_shape[1],self.vocab_size])
        return log_probs


class Bert(Transformer):
    def __init__(self,
                 max_position_embeddings,
                 token_type_vocab_size=2,
                 initializer_range=0.02,
                 with_mlm=False,
                 with_pooler=False,
                 with_cls=False,
                 **kwargs
                 ):
        super(Bert, self).__init__(**kwargs)
        self.with_mlm=with_mlm
        self.with_pooler=with_pooler
        self.with_cls=with_cls

        self.bertEmbedding = BertEmbedding(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            max_position_embeddings=max_position_embeddings,
            token_type_vocab_size=token_type_vocab_size,
            initializer_range=initializer_range,
            dropout=self.hidden_dropout_prob
        )
        self.encoderBlocks=nn.ModuleList([
            EncoderBlock(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                hidden_dropout_prob=self.hidden_dropout_prob,
                intermediate_size=self.intermediate_size,
            ) for _ in range(self.num_hidden_layers)
        ])
        if with_mlm:
            self.masklm=MLM(vocab_size=self.vocab_size,hidden_size=self.hidden_size)
        if with_pooler:
            self.pooled_output=nn.Sequential(
                nn.Linear(self.hidden_size,self.hidden_size),
                nn.Tanh()
            )


    def getCheckpointMap(self):
        mapping = {
                "bert.embeddings.word_embeddings.weight":"bertEmbedding.word_embeddings.weight",
                "bert.embeddings.token_type_embeddings.weight": "bertEmbedding.token_type_embeddings.weight",
                "bert.embeddings.position_embeddings.weight": "bertEmbedding.position_embeddings.weight",
                "bert.embeddings.LayerNorm.weight": "bertEmbedding.norm.weight",
                "bert.embeddings.LayerNorm.bias":"bertEmbedding.norm.bias",
        }
        block={
            ("bert.encoder.layer.{}.attention.self.query.{}", "encoderBlocks.{}.attention.q_kernel.{}"),
            ("bert.encoder.layer.{}.attention.self.key.{}", "encoderBlocks.{}.attention.k_kernel.{}"),
            ("bert.encoder.layer.{}.attention.self.value.{}", "encoderBlocks.{}.attention.v_kernel.{}"),
            ("bert.encoder.layer.{}.attention.output.dense.{}", "encoderBlocks.{}.attention.output_linear.{}"),
            ("bert.encoder.layer.{}.attention.output.LayerNorm.{}",
             "encoderBlocks.{}.attention_resNetSubLayer_input.norm.{}"),
            ("bert.encoder.layer.{}.intermediate.dense.{}", "encoderBlocks.{}.feedForward.ffnc.0.{}"),
            ("bert.encoder.layer.{}.output.dense.{}", "encoderBlocks.{}.feedForward.ffnc.2.{}"),
            ("bert.encoder.layer.{}.output.LayerNorm.{}", "encoderBlocks.{}.ffn_resNetSubLayer_attention.norm.{}"),
        }
        for i in range(self.num_hidden_layers):
            for tuple_ in block:
                for posfix in ['weight','bias']:
                    ori_key=tuple_[0].format(i,posfix)
                    new_key=tuple_[1].format(i,posfix)
                    mapping[ori_key]=new_key

        return mapping



    def forward(self, input_ids,segment_ids):
        input_size=input_ids.size()
        mask=torch.where(input_ids>0,torch.ones(input_size),torch.zeros(input_size))
        x,word_embeddings_table=self.bertEmbedding(input_ids,segment_ids)
        for encoderblock in self.encoderBlocks:
            x=encoderblock.forward(x,mask)
        if self.with_mlm:
            x=self.masklm.forward(x,word_embeddings_table)
        if self.with_pooler:
            if self.with_cls:
                x=x[:,0:1,:].squeeze(dim=1)
            else:
                x=torch.mean(x,dim=1).squeeze(dim=1)
            x=self.pooled_output(x)
        return x

