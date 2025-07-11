from pytorch_pretrained_bert import BertTokenizer
import torch
import numpy as np
import json
import torch.nn.functional as F
from url import seq_padding
from model import Net
import os
import random
seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
torch.backends.cudnn.deterministic = True  # 保证每次调用CuDNN的卷积操作一致
torch.backends.cudnn.benchmark = False

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

tokenizer = BertTokenizer.from_pretrained('../pretrain_model/scibert/vocab.txt', do_lower_case=True)
MAX_LEN = 256
model = Net().cuda()
model.load_state_dict(torch.load('model/model.pkl'))

def read_data(X,author_to_pre30):
    '''
        读取数据
        :param file: 数据路径
        :param temp: 训练or验证
        :return: 作者：兴趣1，兴趣2....
        '''
    data = {}
    with open(X, 'r') as f:
        author_journal_p = json.load(f)
        for item in author_journal_p:
            array = []
            if item['context'] == None:
                continue
            prelables = author_to_pre30[item['author']]
            for label in prelables[:20]:
                array.append((item['context'],label))
            data[item['author']] = array
    return data

def data_process(train_data):
    '''
    处理为bert所需要的输入格式
    :param train_data:
    :return:
    '''
    sentencses1 = ['[CLS] ' + sent[1]  + ' [SEP] ' for sent in train_data]
    sentencses2 = [sent[0] + ' [SEP] ' for sent in train_data]

    tokenized_sents1 = [tokenizer.tokenize(sent) for sent in sentencses1]
    input_ids1 = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents1]

    tokenized_sents2 = [tokenizer.tokenize(sent) for sent in sentencses2]
    input_ids2 = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents2]

    input_ids = [i + j for i, j in zip(input_ids1, input_ids2)]
    input_ids = seq_padding(input_ids, MAX_LEN)

    tokens_type = [[0] * len(i) + [1] * (MAX_LEN - len(i)) for i, j in zip(input_ids1, input_ids2)]

    attention_masks = []

    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    return input_ids, attention_masks, tokens_type


def ttest(model, train_inputs,  train_masks, train_token):
    '''
    预测结果
    :param model: 模型
    :param train_inputs:
    :param train_masks:
    :param train_token:
    :return: 每一个标签的得分
    '''

    logits = model(train_inputs,  train_masks, train_token)
    pre = F.softmax(logits, 1)
    pre = pre.detach().cpu().numpy()
    bert_scores = [pre[x][1] for x in range(len(pre))]
    # print(bert_scores)
    label_score = {a: np.float(b) for a, b in enumerate(bert_scores)}
    return label_score


if __name__ == "__main__":
    author_to_pre30 = {}
    with open("output/recommend_top20.txt", 'r') as f:
        for item in f.readlines():
            item = item.strip().split('\t')
            author_to_pre30[item[0]] = item[1:]

    data = read_data("../raw_data_bkp/dev2.json",author_to_pre30)
    wline = {}
    for n,author in enumerate(list(data.keys())):
        train_inputs,  train_masks, train_token = data_process(data[author])
        train_inputs = torch.tensor(train_inputs).cuda()
        train_masks = torch.tensor(train_masks).cuda()
        train_token = torch.tensor(train_token).cuda()
        label_score = ttest(model, train_inputs,  train_masks, train_token )
        wline[author] = label_score
        if n % 500 == 0:
            print(n)
    print(wline['Stephen S. Intille'])
    with open('output/top20_sort.json','w') as w:
        json.dump(wline, w, ensure_ascii=False)

