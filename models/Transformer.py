import torch
import torch.nn as nn
class Transformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden_size,  # 编码维度,与词向量维度相同,token_type维度相同，位置编码维度相同
                 hidden_dropout_prob,# Dropout比例
                 hidden_act,  # FeedForward隐层的激活函数
                 intermediate_size,  # FeedForward的隐层维度
                 num_attention_heads,  # Attention的头数
                 num_hidden_layers,  # Transformer总层数
                 attention_probs_dropout_prob,
                 **kwargs
                 ):
        super(Transformer,self).__init__()
        self.vocab_size=vocab_size
        self.hidden_size=hidden_size
        self.num_hidden_layers=num_hidden_layers
        self.num_attention_heads=num_attention_heads
        self.attention_head_size=hidden_size//num_attention_heads
        self.intermediate_size=intermediate_size
        self.hidden_dropout_prob=hidden_dropout_prob
        self.hidden_act=hidden_act
        self.attention_probs_dropout_prob=attention_probs_dropout_prob


    def load_weights_from_checkpoint(self,checkpoint_path,parameters_dict,mapping=None):
        """
        根据mapping从checkpoint加载权重
        """
        ori_weights_mapping = torch.load(checkpoint_path)
        for k1, k2 in mapping.items():
            parameters_dict[k2]=ori_weights_mapping.get(k1)
        del ori_weights_mapping


    def mlm(self,input_tensor):
        raise NotImplementedError

