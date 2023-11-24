# coding: UTF-8
import torch
import time
from datetime import timedelta
import csv
import random
PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic

def build_dataset(config):
    id2title = dict()
    id2label = dict()
    id2content = dict()

    index = 0
    with open('./THUCNews/data/Train_DataSet.csv', 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if index > 0:
                try:
                    id2title[row[0]] = row[1]
                    id2content[row[0]] = row[2]
                except:
                    print(index, row)
            index += 1

    with open('./THUCNews/data/Train_DataSet_Label.csv', 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            id = row[0]
            if id in id2title and id in id2content:
                id2label[id] = row[1]

    id2label = random_dic(id2label)
    total_len = len(id2label)
    train_len = int(0.9*total_len)

    train_titles = list()
    train_contents = list()
    train_labels = list()
    test_titles = list()
    test_contents = list()
    test_labels = list()

    index = 0
    for (k, v) in id2label.items():
        if index < train_len:
            train_titles.append(id2title[k])
            train_contents.append(id2content[k])
            train_labels.append(v)
        else:
            test_titles.append(id2title[k])
            test_contents.append(id2content[k])
            test_labels.append(v)
        index += 1

    def load_dataset(titles, contents, labels, pad_size=512):
        outs = []
        for i in range(len(contents)):
            title = titles[i]
            content = contents[i]
            label = labels[i]
            fore_len = int((pad_size-len(title))/2)
            tail_len = len(content) - fore_len + 2
            token = title + content[0:fore_len] + content[tail_len:-1]
            token = config.tokenizer.tokenize(token)
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token)

            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids += ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            outs.append((token_ids, int(label), seq_len, mask))
        return outs
    train = load_dataset(train_titles, train_contents, train_labels, config.pad_size)
    test = load_dataset(test_titles, test_contents, test_labels,  config.pad_size)
    #dev = load_dataset(config.dev_path, config.pad_size)

    print(f'train set:{len(train)}, test set:{len(test)}')
    return train, test #,dev


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
