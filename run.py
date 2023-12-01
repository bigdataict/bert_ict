# coding: UTF-8
import os
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, help='choose a model: Bert, ERNIE', default='bert')
parser.add_argument('--dataset', type=str, help='choose a dataset', default='THUCNews')
parser.add_argument('--out_dir', type=str, help='output dir', default='./THUCNews')
parser.add_argument('--learning_rate', type=float,help='learning rate?', default=2e-5)
parser.add_argument('--pretrain', type=str, help='choose a pretrain model', default='bert_pretrain')
parser.add_argument('--batch_size', type=int, help='mini-batch', default=16)
parser.add_argument('--k_fold', type=int, help='k_fold', default=10)
args = parser.parse_args()

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

    print("Loading data...")
    trains, tests = build_dataset(config)
    for i in range(config.k_fold):
        train_iter = build_iterator(trains[i], config)
        test_iter = build_iterator(tests[i], config)
        # train
        print("No: "+str(i)+" fold")
        model = x.Model(config).to(config.device)
        train(config, model, train_iter, test_iter, i)
