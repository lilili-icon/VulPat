import os
import pandas as pd
import pickle
import csv
from more_itertools import chunked
import numpy as np
from pycparser import c_parser
import re

parser = c_parser.CParser()
def write2txt(file,data):
    if not os.path.exists(file):
        with open(file, mode='a', encoding='utf-8') as ff:
            header = [['CVE','CWE','func','ID']]
            df = pd.DataFrame(data=header)
            df.to_csv(file, header=False, index=False, mode='a')
            print("文件创建成功！")
    df = pd.DataFrame(data=data)
    df.to_csv(file, header=False, index=False, mode='a')

def enableParse(code):
    '''
    判断代码是否可以解析
    :param code:
    :return:
    '''
    try:
        parser.parse(code)
        return True
    except Exception as e:
        pass
    return False

def read4chunk(file,test_batch_size):
    '''
    将文件按块读取
    :param file:
    :param test_batch_size:
    :return:
    '''
    csv.field_size_limit(500 * 1024 * 1024)
    batched_data = pd.read_csv(file,chunksize=test_batch_size )

    return batched_data

def rightFile(input,output):
    '''

    :param input:
    :param output:
    :return:
    '''
    right = 0
    chunks = read4chunk(input,1000)
    for batch_data in chunks:
        for _,item in batch_data.iterrows():
            if enableParse(item['func_before']):
                right = right+1
                item['ID'] = right
                chunk_source_list = [item]
                write2txt(output,chunk_source_list)
    return

def groupCWE(file):
    df = pd.read_csv(file)
    print('------------')
    feature = []
    for index,example in df.iterrows()  :
        feature.append(example['CWE'])
    uni_feature = set(feature)

    for feature in uni_feature:
        a = df.query('CWE==@feature')
        path  = '../data/CWE/'+str(a.shape[0])+'-'+str(feature)+'.csv'
        if not os.path.exists(path):
            write2txt(path,a)


def toPkl(url):
    files = os.listdir(url)
    for file in files:
        path = url+'/'+str(file)
        df = pd.read_csv(path, encoding='utf-8')
        df.to_pickle('../data/CWE-pkl/'+file+'.pkl')

def code2node(input_file):
    df = pd.read_csv("../data3/"+input_file, encoding='utf-8')
    df.to_pickle("../data3/test/pkl/"+input_file + '.pkl')
    trees = pd.read_pickle("../data3/test/pkl/"+input_file + '.pkl')

    from prepare_data import get_sequences

    def trans_to_sequences(ast):
        sequence = []
        try:
            get_sequences(ast, sequence)
        except Exception as e:
            pass
        return sequence
    print()
    corpus = trees['func'].apply(parser.parse).apply(trans_to_sequences)
    str_corpus = [' '.join(c) for c in corpus]
    trees['func'] = pd.Series(str_corpus)
    trees.to_csv("../data3/output/"+input_file+'.tsv')
    return

def clear(path,output):
    chunks = read4chunk(path, 1000)
    for batch_data in chunks:
        for _, item in batch_data.iterrows():
            func = str(item['func'])
            if len(func)==0:
                continue
            pat = '[A-Z][A-Za-z]*'

            lst = re.findall(pat, func)

            item['func'] = lst

            df = pd.DataFrame(data=[item])
            df.to_csv(output, header=False, index=False, mode='a')


    return

def split_data(file,ratio):
    path = "../data/train/train3/" + file
    data = pd.read_csv(path)
    data_num = len(data)
    ratios = [int(r) for r in ratio.split(':')]
    train_split = int(ratios[0]/sum(ratios)*data_num)
    data = data.sample(frac=1, random_state=666)
    train = data.iloc[:train_split]
    test = data.iloc[train_split:]

    def check_or_create(path):
        if not os.path.exists(path):
            os.mkdir(path)
    train_path = "../../train/train3/"
    check_or_create(train_path)
    train.to_csv(train_path+"train3.txt",header=False,mode='a')


    test_path = "../../train/train3/"
    check_or_create(test_path)
    test.to_csv(test_path +"val3.txt",header=False,mode='a')

def get_pattern(file):


    return
if __name__ == '__main__':
    #code2node("test.txt")

    clear("../data3/output/test.txt.tsv","../data3/output/clear/test.tsv")
    # files = os.listdir("../data/train/train3")
    # # #
    # for file in files:
    #
    #     split_data(file,"3:1")
    # #





