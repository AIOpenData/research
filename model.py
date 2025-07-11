from torch import nn
import torch.nn as nn
from pytorch_pretrained_bert import BertModel,BertTokenizer
import torch

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('../pretrain_model/scibert/scibert.tar.gz')#
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(768, 2)# 二分类问题

    def forward(self, context, mask, pos):
        # context = a  # 输入的句子
        # mask = b  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, token_type_ids=pos)
        out = self.fc(pooled)
        return out
