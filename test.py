import time
import torch
import numpy as np
from importlib import import_module
import argparse
import csv
import os

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE', default='bert')
parser.add_argument('--dataset', type=str, help='choose a dataset', default='THUCNews')
parser.add_argument('--out_dir', type=str, help='output dir', default='./THUCNews')
args = parser.parse_args()

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def load_dataset(contents, ids, pad_size=32):
    outs = []
    for i in range(len(contents)):
        content = contents[i]
        id = ids[i]
        token = config.tokenizer.tokenize(content)
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
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    ids = []
    titles = []

    index = 0
    with open('./THUCNews/data/Test_DataSet.csv', 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if index > 0:
                try:
                    titles.append(row[1])
                    ids.append(row[0])
                except:
                    print(index, row)
            index += 1
    test = load_dataset(titles, ids, config.pad_size)
    print(len(test))
    test_iter = build_iterator(test, config)

    submit = [['id', 'label']]
    # train
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    with torch.no_grad():
        for (title, id) in test_iter:
            outputs = model(title)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            submit.append([id[0], predic[0]])

    # python2可以用file替代open
    outfile = os.path.join(args.out_dir, "test.csv")
    with open(outfile, "a", encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in submit:
            writer.writerow(row)
