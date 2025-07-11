
import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
from sklearn import metrics
import torch.nn.functional as F
from model import Net
import random
from loguru import logger
from torch.utils.data import DataLoader,TensorDataset,RandomSampler,SequentialSampler
import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
logger.add('log/train_{time}.log')

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
torch.backends.cudnn.deterministic = True  # 保证每次调用CuDNN的卷积操作一致
torch.backends.cudnn.benchmark = False

MAX_LEN = 256

def seq_padding(X,ML):
    '''
    把每一句话处理为等长序列
    :param X: 输入句子列表
    :param ML: 设置的最大长度
    :return:
    '''
    padding = 0
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x[:ML] for x in X
    ])

def read_data(file, temp):
    '''
    读取数据
    :param file: 数据路径
    :param temp: 训练or验证
    :return: （论文集，兴趣点，标签）
    '''

    train_data = []
    with open('../raw_data_bkp/labels.txt') as f:
        tags = [item.strip() for item in f.readlines()]

    with open(file, 'r') as f:
    
        if temp:
            author_journal_p = json.load(f)[:-100]
        else:
            author_journal_p = json.load(f)[-100:]
        for n, item in enumerate(author_journal_p):
            labels = item['label']
            if item['context'] == None:
                continue
            for label in labels:
                train_data.append((item['context'], label, 1))
            for _ in range(15):
                tag = tags[random.randint(0, len(tags) - 1)]
                if tag not in labels:
                    train_data.append((item['context'], tag, 0))

    return train_data

def data_process(train_data):
    '''
    处理为bert所需要的输入格式
    :param train_data:
    :return:
    '''

    sentencses1 = ['[CLS] ' + sent[1] + ' [SEP] ' for sent in train_data]

    sentencses2 = [sent[0] + ' [SEP] ' for sent in train_data]

    labels = [label[2] for label in train_data]

    tokenizer = BertTokenizer.from_pretrained('../pretrain_model/scibert/vocab.txt', do_lower_case=True)

    tokenized_sents1 = [tokenizer.tokenize(sent) for sent in sentencses1]
    input_ids1 = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents1]

    tokenized_sents2 = [tokenizer.tokenize(sent) for sent in sentencses2]
    input_ids2 = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents2]
    # 对论文和兴趣点进行拼接
    input_ids = [i+j for i, j in zip(input_ids1, input_ids2)]
    input_ids = seq_padding(input_ids, MAX_LEN)
    # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
    tokens_type = [[0]*len(i)+[1]*(MAX_LEN-len(i)) for i, j in zip(input_ids1, input_ids2)]

    attention_masks = []

    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    return input_ids, labels, attention_masks, tokens_type


def evaluate(model, validation_dataloade):
    '''
    验证
    :param model: 模型
    :param validation_dataloade: 验证数据
    :return: ACC,LOSS
    '''
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for step, batch in enumerate(validation_dataloade):
            batch = tuple(t.to('cuda') for t in batch)
            b_input_ids, b_input_mask, b_labels, b_token = batch
            outputs = model(b_input_ids, b_input_mask, b_token)
            loss = F.cross_entropy(outputs,b_labels)
            loss_total += loss
            labels = b_labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    return acc, loss_total / len(validation_dataloade)

def train(model,train_dataloader):
    '''

    :param model: 模型
    :param train_dataloader: 训练数据
    :return: 最终的模型
    '''
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    total_batch,best_dev = 0,0
    for item in range(10):
        # 训练开始
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to('cuda') for t in batch)
            b_input_ids, b_input_mask, b_labels , b_token = batch#, b_token
            outputs = model(b_input_ids, b_input_mask, b_token)
            model.zero_grad()
            loss = F.cross_entropy(outputs, b_labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                true = b_labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(model, validation_dataloader)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}'
                logger.info(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc))
                if best_dev <= dev_acc:
                    best_dev = dev_acc
                    torch.save(model.state_dict(), "model/model.pkl")

            total_batch += 1

if __name__ == "__main__":

    train_data = read_data("../raw_data_bkp/train2.json", temp=True)
    dev_data = read_data("../raw_data_bkp/train2.json", temp=False)
    logger.info('data load end!')
    train_inputs, train_labels, train_masks,train_token = data_process(train_data)
    validation_inputs, validation_labels, validation_masks,validation_token = data_process(dev_data)
    logger.info('data process end!')
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)
    train_token = torch.tensor(train_token)
    validation_token = torch.tensor(validation_token)
    logger.info('data random end!')
    #生成dataloader
    batch_size = 32
    train_data = TensorDataset(train_inputs, train_masks, train_labels,train_token)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels,validation_token)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    model = Net()
    model = model.cuda()
    train(model,train_dataloader)
