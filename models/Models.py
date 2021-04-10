# -*- coding:utf-8 -*-
import torch.nn as nn
import json
import torch
from bertSpytorch.models.Bert import Bert

def build_transformer_model(
        config_path=None,
        checkpoint_path=None,
        model='bert',
        **kwargs
    ):
    """根据配置文件构建模型，可选加载checkpoint权重"""
    configs={}
    if config_path is not None:
        configs.update(json.load(open(config_path)))
    configs.update(kwargs)
    models={
        'bert':Bert,
    }
    if isinstance(model,str):
        model=model.lower()
        MODEL=models[model]
    else:
        MODEL=model

    transfomer=MODEL(**configs)

    if checkpoint_path is not None:
        parameters_dict={k:_ for k,_ in transfomer.state_dict().items()}
        transfomer.load_weights_from_checkpoint(checkpoint_path,parameters_dict,transfomer.getCheckpointMap())
        transfomer.load_state_dict(parameters_dict)
    return transfomer
