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

def build_dataset(config,fullText):
    id2title = dict()
    id2label = dict()
    id2content = dict()
    k_fold = config.k_fold

    index = 0
    with open('./THUCNews/data/Train_DataSet.csv', 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if index > 0:
                try:
                    id2title[row[0]] = row[1]
                    id2content[row[0]] = row[2]
                except:
                    pass
            index += 1

    with open('./THUCNews/data/Train_DataSet_Label.csv', 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            id = row[0]
            if id in id2title and id in id2content:
                id2label[id] = row[1]

    id2label = random_dic(id2label)
    test_len = int(len(id2label)/k_fold)

    trainsets_title = []
    trainsets_content = []
    trainsets_label = []
    testsets_title = []
    testsets_content = []
    testsets_label = []


    test_titles = list()
    test_contents = list()
    test_labels = list()

    flag = 0
    for (k, v) in id2label.items():
        if flag <= test_len:
            test_titles.append(id2title[k])
            test_contents.append(id2content[k])
            test_labels.append(v)
        else:
            testsets_title.append(test_titles)
            testsets_content.append(test_contents)
            testsets_label.append(test_labels)
            test_titles = list()
            test_contents = list()
            test_labels = list()
            flag = 0
            test_titles.append(id2title[k])
            test_contents.append(id2content[k])
            test_labels.append(v)
        flag += 1
    testsets_title.append(test_titles)
    testsets_content.append(test_contents)
    testsets_label.append(test_labels)
    print("len k_datasets:", len(testsets_title))
    for i in range(k_fold):
        train_titles = list()
        train_contents = list()
        train_labels = list()
        for j in range(k_fold):
            if j == i:
                print("No:",i,"test len:",len(testsets_title[i]))
                continue
            train_titles += testsets_title[j]
            train_contents += testsets_content[j]
            train_labels += testsets_label[j]
        trainsets_title.append(train_titles)
        trainsets_content.append(train_contents)
        trainsets_label.append(train_labels)
        print("trainset len:",len(train_titles))


    def load_dataset(titles, contents, labels, pad_size=512):
        outs = []
        num0 = 0
        num1 = 0
        num2 = 0
        for i in range(len(contents)):
            title = titles[i]
            content = contents[i]
            label = labels[i]
            fore_len = int((pad_size-len(title)))
            token = title + content[0:fore_len]
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
            if label == '0': num0 += 1
            if label == '1': num1 += 1
            if label == '2': num2 += 1
        print(f"ratio {num0} : {num1} : {num2}")
        return outs
    trains = []
    tests = []
    for i in range(k_fold):
        train = load_dataset(trainsets_title[i], trainsets_content[i], trainsets_label[i], config.pad_size)
        test = load_dataset(testsets_title[i], testsets_content[i], testsets_label[i],  config.pad_size)
        trains.append(train)
        tests.append(test)

    return trains, tests #,dev


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
