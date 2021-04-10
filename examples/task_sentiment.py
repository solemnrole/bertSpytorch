# -*- coding:utf-8 -*-
'''
情感分析小样本学习
借鉴苏剑林。
https://spaces.ac.cn/archives/7764
'''
import torch
from bertSpytorch.models import build_transformer_model
from bertSpytorch.utils.snippets import sequence_padding, DataGenerator
from bertSpytorch.utils.snippets import open
from bertSpytorch.utils.tokenizers import Tokenizer
import numpy as np
import argparse
import os

parser=argparse.ArgumentParser()
parser.add_argument("-td","--train_data",default='D:/datasets/sentiment/sentiment.train.data',help="数据路径")
parser.add_argument("-vd","--valid_data",default='D:/datasets/sentiment/sentiment.valid.data',help="数据路径")
parser.add_argument("-tsd","--test_data",default='D:/datasets/sentiment/sentiment.test.data',help="数据路径")
parser.add_argument("-cf","--config",default='D:/MODEL/chinese_roberta_wwm_ext_pytorch/bert_config.json',help="参数配置文件")
parser.add_argument("-v","--vocab",default='D:/MODEL/chinese_roberta_wwm_ext_pytorch/vocab.txt',help="参数配置文件")
parser.add_argument("-cp","--checkpoint",default='D:/MODEL/chinese_roberta_wwm_ext_pytorch/pytorch_model.bin',help="预训练模型路径")
parser.add_argument("-m","--model",default='bert',help="选择的基础模型,其他模型开发中~")
parser.add_argument("-seed","--seed",default=3243,help="随机数")
parser.add_argument("-sd","--save_dir",default='snapshot',help="模型存储路径")
parser.add_argument("-hidden_size","--hidden_size",default=768,help="分类阶段隐藏层")
parser.add_argument("-class_drop_rate","--class_drop_rate",default=0.2,help="分类阶段droprate")
parser.add_argument("-prefix","--prefix",default='很满意',help="前缀")
parser.add_argument("-mask_idx","--mask_idx",default=1,help="分类标志位，bert有[CLS]标志")
parser.add_argument("-maxlen","--maxlen",default=128,help="句子长度")
parser.add_argument("-batch_size","--batch_size",default=32,help="batchsize")
parser.add_argument("-lr","--lr",default=1e-5,help="学习率")
parser.add_argument("-lr_decay","--lr_decay",default=0.9,help="")
parser.add_argument("-min_lr","--min_lr",default=1e-6,help="")
parser.add_argument("-epochs","--epochs",default=1000,help="epochs")
parser.add_argument("-gpu","--gpu",default=True,help="是否使用GPU")
parser.add_argument("-device","--device",default=0,help="")
args=parser.parse_args()

# 建立分词器
tokenizer = Tokenizer(args.vocab, do_lower_case=True)


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text, label = l.strip().split('\t')
            D.append((text, int(label)))
    return D


class data_generator(DataGenerator):
    """数据生成器
    """
    def __init__(self,args,data):
        super(data_generator,self).__init__(batch_size=args.batch_size,data=data)
        self.args=args

    def __iter__(self, random=True):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text, label) in self.sample(random):

            token_ids, segment_ids = tokenizer.encode(text, maxlen=self.args.maxlen)
            source_ids= token_ids[:]
            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(label)

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)

                yield  torch.tensor(batch_token_ids,dtype=torch.int64), \
                       torch.tensor(batch_segment_ids,dtype=torch.int64), \
                       torch.tensor(batch_output_ids,dtype=torch.int64)
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []

class ClassfierModel(torch.nn.Module):
    def __init__(self,baseModel,hidden_size,drop_out):
        super(ClassfierModel,self).__init__()
        self.baseModel=baseModel
        self.class_liner=torch.nn.Sequential(
            torch.nn.Linear(hidden_size,1),
            torch.nn.Dropout(drop_out),
            torch.nn.Sigmoid()
        )

    def forward(self, input_ids,segment_ids):
        pooler_logits=self.baseModel(input_ids,segment_ids)
        logits=self.class_liner(pooler_logits).squeeze(dim=-1)
        return logits

def myloss(logits,batch_output_ids):
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,batch_output_ids.float())
    return loss

def evaluate(model,data):
    model.eval()
    predict_right_num=0
    all_num=0
    gold_num=0

    for batch_token_ids, batch_segment_ids, batch_output_ids in data:
        logits_shape=batch_output_ids.size()
        ones,zeros=torch.ones(logits_shape), torch.empty(logits_shape).random_(2,3)
        if args.cuda:
            batch_token_ids, batch_segment_ids, batch_output_ids = \
                batch_token_ids.cuda(), \
                batch_segment_ids.cuda(), \
                batch_output_ids.cuda()
            ones,zeros=ones.cuda(),zeros.cuda()

        logits = model.forward(input_ids=batch_token_ids, segment_ids=batch_segment_ids)
        logits_=torch.where(logits>0.6,ones,zeros)
        predict_right_num+=torch.sum(torch.eq(logits_,batch_output_ids))
        gold_num+=torch.sum(torch.eq(batch_output_ids,1))
        all_num+=torch.sum(torch.eq(logits_,1))
    p=predict_right_num/(all_num+1e-5)
    r=predict_right_num/(gold_num+1e-5)
    f1=2*p*r/(p+r+1e-5)
    return p,r,f1


def main(args):
    if not torch.cuda.is_available() or not args.gpu:
        args.cuda=False
        args.device=None
        torch.manual_seed(args.seed)
    else:
        args.cuda = True

    basemodel=build_transformer_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        model=args.model,
        with_mlm=False,
        with_pooler=True,
        with_cls=False,
    )
    model=ClassfierModel(
        baseModel=basemodel,
        hidden_size=args.hidden_size,
        drop_out=args.class_drop_rate
    )
    if args.cuda:
        torch.cuda.set_device(args.device)
        torch.cuda.manual_seed(args.seed)
        model.cuda()

    train_data=load_data(args.train_data)
    valid_data=load_data(args.valid_data)
    test_data=load_data(args.test_data)

    #测试使用
    train_data=train_data[:64]
    valid_data=valid_data[:64]
    test_data=test_data[:64]

    # 转换数据集
    train_generator = data_generator(args=args,data=train_data)
    valid_generator = data_generator(args=args,data=valid_data)
    test_generator = data_generator(args=args,data=test_data)

    #定义优化器
    optimizer=torch.optim.Adam(params=model.parameters(),lr=args.lr)
    optimizer.zero_grad()

    best_score=0.
    # 模型保存位置
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, 'sentiment_{}_best.pth'.format(args.model))

    for epo in range(args.epochs):
        idx=0
        model.train()
        for batch_token_ids, batch_segment_ids, batch_output_ids in train_generator:
            if args.cuda:
                batch_token_ids, batch_segment_ids,batch_output_ids=\
                                                    batch_token_ids.cuda(),\
                                                    batch_segment_ids.cuda(),\
                                                    batch_output_ids.cuda()

            logits=model.forward(input_ids=batch_token_ids,segment_ids=batch_segment_ids)
            loss=myloss(logits=logits,batch_output_ids=batch_output_ids)

            #更新指标
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 显示指标
            print('[{}, {}] batchsize:{}, lr:{}, mean_loss: {:.5f}'.format(
                epo + 1,
                idx,
                len(batch_token_ids),
                optimizer.state_dict().get('param_groups')[0].get('lr'),
                loss.item(),
                )
            )
            idx += 1
        #每个epoch结束，计算在验证集上的acc值，
        p,r,f1=evaluate(model,valid_generator)
        print('epoch:{}, p:{},r:{},f1:{}'.format(epo,p,r,f1))
        if f1>best_score:
            best_score=f1
            checkpoint={
                "state_dict":model.state_dict(),
                "epoch":epo,
                "optimizer":optimizer.state_dict(),
                "best_score":best_score
            }
            torch.save(checkpoint,save_path)
            print('Best tmp model score:{}'.format(best_score))
        if f1<best_score:
            model.load_state_dict(torch.load(save_path))
            for p in optimizer.param_groups:
                p['lr'] *= args.lr_decay
        n_lr = optimizer.state_dict().get('param_groups')[0].get('lr')
        if n_lr<=args.min_lr:
            print('earlystopping~')
            break
    p,r,f1=evaluate(model,test_generator)
    print('测试集 p:{},r:{},f1:{}'.format(p,r,f1))

if __name__=='__main__':
    main(args)