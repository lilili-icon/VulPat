import pandas as pd
import random
import torch
import time
import numpy as np
from gensim.models.word2vec import Word2Vec
from model import BatchProgramClassifier
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import sys
import csv


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    data, labels = [], []
    for _, item in tmp.iterrows():
        data.append(item[2])
        #labels.append(item[2])
        labels.append(item[3])
    return data, torch.LongTensor(labels)

def get_Tensor(data,label,file):
    dataList = data.tolist()
    labelList = label.tolist()
    with open(file,encoding='utf-8',mode='a', newline ='') as f:
        writer = csv.writer(f)
        for label, data in zip(labelList, dataList):
            data.append(label)
            writer.writerow(data)

if __name__ == '__main__':
    root = 'data/'
    train_data = pd.read_pickle(root+'train/4080-CWE-119AST/blocks.pkl')
    test_data = pd.read_pickle(root+'test/4080-CWE-119AST/blocks.pkl')

    word2vec = Word2Vec.load(root+"train/4080-CWE-119AST/embedding/node_w2v_128").wv
    embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 104
    EPOCHS = 1
    BATCH_SIZE = 64
    USE_GPU = False
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]

    model = BatchProgramClassifier(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                   USE_GPU, embeddings)
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.CrossEntropyLoss()

    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []
    best_acc = 0.0
    print('Start training...')
    # training procedure
    best_model = model
    for epoch in range(EPOCHS):

        i = 0
        while i < len(train_data):
            batch = get_batch(train_data, i, BATCH_SIZE)
            i += BATCH_SIZE
            train_inputs, train_labels = batch
            if USE_GPU:
                train_inputs, train_labels = train_inputs, train_labels.cuda()

            model.zero_grad()
            model.batch_size = len(train_labels)
            model.hidden = model.init_hidden()
            output = model(train_inputs)
            file = "embedding/CWE-119.csv"
            data = get_Tensor(output,train_labels,file)

        print("写入完成")
