# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):
    """配置参数"""

    def __init__(self, args):
        self.model_name = 'bert'
        self.save_path = args.out_dir + '/saved_dict/'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.k_fold = args.k_fold
        self.num_classes = 3
        self.num_epochs = 7  # epoch数
        self.batch_size = args.batch_size  # mini-batch大小
        self.pad_size = 512  # 每句话处理成的长度(短填长切)
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.learning_rate = args.learning_rate  # 学习率
        self.bert_path = './'+args.pretrain
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out
