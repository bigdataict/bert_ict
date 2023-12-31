import time
import torch
import numpy as np
from importlib import import_module
import argparse
import csv
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, help='choose a model: Bert, ERNIE', default='bert')
parser.add_argument('--dataset', type=str, help='choose a dataset', default='THUCNews')
parser.add_argument('--out_dir', type=str, help='output dir', default='./THUCNews')
parser.add_argument('--learning_rate', type=float, help='learning rate?', default=2e-5)
parser.add_argument('--pretrain', type=str, help='choose a pretrain model', default='bert_pretrain')
parser.add_argument('--batch_size', type=int, help='mini-batch', default=16)
parser.add_argument('--k_fold', type=int, help='k_fold', default=10)
args = parser.parse_args()

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def load_dataset(titles, contents, ids, pad_size=512):
    outs = []
    for i in range(len(contents)):
        title = titles[i]
        content = contents[i]
        id = ids[i]
        fore_len = int((pad_size - len(title)))
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
        outs.append((token_ids, id, seq_len, mask))
    return outs


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = 1
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = [_[1] for _ in datas]

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
    iter = DatasetIterater(dataset, 1, config.device)
    return iter


if __name__ == '__main__':
    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(args)
    if os.path.exists(args.out_dir + '/saved_dict') is False:
        os.makedirs(args.out_dir + '/saved_dict')

    ids = []
    titles = []
    contents = []

    index = 0
    with open('./THUCNews/data/Test_DataSet.csv', 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if index > 0:
                try:
                    contents.append(row[2])
                    titles.append(row[1])
                    ids.append(row[0])
                except:
                    titles.append(row[1])
                    ids.append(row[0])
                    contents.append("")
            index += 1
    test = load_dataset(titles, contents, ids, config.pad_size)
    # train
    model = x.Model(config).to(config.device)


    def evaluate(model, iter, model_name):
        submit = [['id', 'label']]
        model.load_state_dict(torch.load(config.save_path + model_name + '.ckpt'))
        model.eval()
        with torch.no_grad():
            for (title, id) in iter:
                outputs = model(title)
                predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                submit.append([id[0], predic[0]])

        outfile = os.path.join(args.out_dir, model_name + ".csv")
        with open(outfile, "a", encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in submit:
                writer.writerow(row)


    test_iter = build_iterator(test, config)
    for i in range(config.k_fold):
        evaluate(model, test_iter, str(i))

    # k fold vote
    id = []
    pred = []

    dic = dict()
    for i in range(config.k_fold):
        index = 0
        flag = True
        with open(args.out_dir + '/' + str(i) + '.csv', 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if flag:
                    flag = False
                    continue
                if i == 0:
                    id.append(row[0])
                    pred.append([row[1]])
                else:
                    if id[index] == row[0]:
                        pred[index].append(row[1])
                    else:
                        print("error")
                index += 1

    data = []

    for i in range(len(id)):
        li = pred[i]
        p = max(li, key=li.count)
        data.append([id[i], p])

    with open(args.out_dir + '/' + 'pseu_'+args.pretrain+'.csv', "a", encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label'])
        for row in data:
            writer.writerow(row)
