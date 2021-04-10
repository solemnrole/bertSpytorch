# -*- coding:utf-8 -*-
import torch
from bert4pytorch.models import build_transformer_model


class ModelTest():
    def test(self):
        model=build_transformer_model(
            config_path='D:/MODEL/chinese_L-12_H-768_A-12/bert_config.json',
            checkpoint_path='D:/MODEL/chinese_roberta_wwm_ext_pytorch/pytorch_model.bin',
            model='bert',
            with_mlm=False,
        )
        import random
        input_ids=torch.tensor([[102]+random.sample(range(104,21128),510)+[103],
                                [102]+random.sample(range(104,21128),510)+[103]])
        token_type_ids=torch.ones((2,512),dtype=torch.int64)
        model(input_ids,token_type_ids)