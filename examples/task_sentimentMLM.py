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
parser.add_argument("-train_frac","--train_frac",default=0.01,help="未标注数据占比")
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

def random_masking(tokenizer,token_ids):
    """对输入进行随机mask
    """
    rands = np.random.random(len(token_ids))
    source, target = [], []
    for r, t in zip(rands, token_ids):
        if r < 0.15 * 0.8:
            source.append(tokenizer._token_mask_id)
            target.append(t)
        elif r < 0.15 * 0.9:
            source.append(t)
            target.append(t)
        elif r < 0.15:
            source.append(np.random.choice(tokenizer._vocab_size - 1) + 1)
            target.append(t)
        else:
            source.append(t)
            target.append(0)
    return source, target


class data_generator(DataGenerator):
    """数据生成器
    """
    def __init__(self,data):
        super(data_generator,self).__init__(batch_size=args.batch_size,data=data)

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text, label) in self.sample(random):
            if label != 2:
                text = args.prefix + text
            token_ids, segment_ids = tokenizer.encode(text, maxlen=args.maxlen)
            if random:
                source_ids, target_ids = random_masking(token_ids)
            else:
                source_ids, target_ids = token_ids[:], token_ids[:]
            if label == 0:
                source_ids[args.mask_idx] = tokenizer._token_mask_id
                target_ids[args.mask_idx] = args.neg_id
            elif label == 1:
                source_ids[args.mask_idx] = tokenizer._token_mask_id
                target_ids[args.mask_idx] = args.pos_id

            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                yield  torch.tensor(batch_token_ids,dtype=torch.int64), \
                       torch.tensor(batch_segment_ids,dtype=torch.int64), \
                       torch.tensor(batch_output_ids,dtype=torch.int64)
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []



def myloss(logits,batch_output_ids):
    mask=torch.ne(batch_output_ids,0)
    one_hot_=torch.zeros(logits.size()).scatter_(dim=2,
                                               index=batch_output_ids.unsqueeze(dim=-1),
                                               value=1)
    per_example_loss=-torch.sum(torch.mul(logits,one_hot_),dim=-1)*mask
    loss=torch.sum(per_example_loss)/(torch.sum(mask)+1e-5)
    return loss

def evaluate(model,data):
    model.eval()
    predict_right_num=0
    all_num=0
    for batch_token_ids, batch_segment_ids, batch_output_ids in data:
        batch_size=batch_output_ids.size(0)
        pos_ids=torch.empty(batch_size).random_(args.pos_id,args.pos_id+1)
        neg_ids=torch.empty(batch_size).random_(1)
        if args.cuda:
            batch_token_ids, batch_segment_ids, batch_output_ids = \
                batch_token_ids.cuda(), \
                batch_segment_ids.cuda(), \
                batch_output_ids.cuda()
            pos_ids,neg_ids=pos_ids.cuda(),neg_ids.cuda()

        logits = model.forward(input_ids=batch_token_ids, segment_ids=batch_segment_ids)
        logits=logits[:, args.mask_idx, [args.pos_id, args.neg_id]]
        logits_ids=torch.where(logits[:,0]>=logits[:,1],pos_ids,neg_ids)
        token_ids_true=batch_output_ids[:,args.mask_idx]
        predict_right_num+=torch.sum(torch.eq(logits_ids,token_ids_true))
        all_num+=torch.sum(torch.eq(token_ids_true,pos_ids))
    return predict_right_num/(all_num+1e-5)




def main():
    if not torch.cuda.is_available() or not args.gpu:
        args.cuda=False
        args.device=None
        torch.manual_seed(args.seed)
    else:
        args.cuda = True

    model=build_transformer_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        model=args.model,
        with_mlm=True,
    )
    if args.cuda:
        torch.cuda.set_device(args.device)
        torch.cuda.manual_seed(args.seed)
        model.cuda()

    train_data=load_data(args.train_data)
    valid_data=load_data(args.valid_data)
    test_data=load_data(args.test_data)

    #模拟标注数据和非标注数据
    num_labeled=int(len(train_data)*args.train_frac)
    unlabeled_data=[(t,2) for t,l in train_data[num_labeled:]]
    train_data=train_data[:num_labeled]
    train_data=unlabeled_data+train_data

    # 测试使用
    train_data = train_data[:64]
    valid_data = valid_data[:64]
    test_data = test_data[:64]


    pos_id = tokenizer.token_to_id(u'很')
    neg_id = tokenizer.token_to_id(u'不')

    args.pos_id=pos_id
    args.neg_id=neg_id

    # 转换数据集
    train_generator = data_generator(data=train_data)
    valid_generator = data_generator(data=valid_data)
    test_generator = data_generator(data=test_data)

    #定义优化器
    optimizer=torch.optim.Adam(params=model.parameters(),lr=args.lr)
    optimizer.zero_grad()

    best_acc=0.
    # 模型保存位置
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, 'sentimentMLM_{}_best.pth'.format(args.model))

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
        acc=evaluate(model,valid_generator)
        if acc>best_acc:
            best_acc=acc
            checkpoint={
                "state_dict":model.state_dict(),
                "epoch":epo,
                "optimizer":optimizer.state_dict(),
                "best_acc":best_acc
            }
            torch.save(checkpoint,save_path)
            print('Best tmp model acc:{}'.format(best_acc))
        if acc<best_acc:
            model.load_state_dict(torch.load(save_path))
            for p in optimizer.param_groups:
                p['lr'] *= args.lr_decay
        n_lr = optimizer.state_dict().get('param_groups')[0].get('lr')
        if n_lr<=args.min_lr:
            print('earlystopping~')
            break
    acc=evaluate(model,test_generator)
    print('测试集acc:',acc)

if __name__=='__main__':
    main()



