import os
import pandas as pd
import pickle
import csv
from more_itertools import chunked
import numpy as np
import torch


from pycparser import c_parser
parser = c_parser.CParser()

def parse(source):
    file = "error.txt"
    try:
        parser.parse(source)
    except Exception as e:
        pass

def parse(test_batch_size=1000):
    csv.field_size_limit(500 * 1024 * 1024)
    with open('allData_2.csv', encoding='UTF-8') as f:
        data = csv.reader(f)
        batched_data = chunked(data, test_batch_size)
        right = 1
        error = 1
        chunk_source_list = [["ID", "CVE", "CWE", "func"]]
        df = pd.DataFrame(data=chunk_source_list)
        file = "right.txt"
        df.to_csv(file, header=False, index=False, mode='a')
        print("start parser")
        for batch_data in batched_data:
            for CVE_ID,CWE_D,func_before in batch_data:
                try:
                    parser.parse(func_before)
                    chunk_source_list = [[right,CVE_ID, CWE_D, func_before]]
                    df = pd.DataFrame(data=chunk_source_list)
                    file = "right.txt"
                    df.to_csv(file, header=False, index=False, mode='a')
                    right =right + 1
                except Exception as e:
                    chunk_source_list = [[error,CVE_ID, CWE_D, func_before,e]]
                    df = pd.DataFrame(data=chunk_source_list)
                    file = "error.txt"
                    df.to_csv(file, header=False, index=False, mode='a')
                    error = error + 1
                    pass
                continue


def group():
    df = pd.read_csv("right.txt")
    print('------------')
    feature = []
    for index,example in df.iterrows()  :
        feature.append(example[2])
    uni_feature = set(feature)

    for feature in uni_feature:
        a = df.query('CWE==@feature')
        b = pd.DataFrame(data=a)
        path  = 'data/'+str(feature)+'.txt'
        if not os.path.exists(path):
            df.to_pickle('data.pkl')
            #b.to_csv(path, header=True, index=False, mode='a')
        print('------------')
    #print(feature)
    print(uni_feature)

    print('------------')
def scaner_file (url):
    file = os.listdir(url)
    print(file)

# def toPkl(url):
#     files = os.listdir(url)
#     for file in files:
#         file = url+'/'+str(file)
#         df = pd.read_csv(file, encoding='utf-8')
#         df.to_pickle('save/'+file+'.pkl')

def toPkl(url):

    df = pd.read_csv(url, encoding='utf-8')
    df.to_pickle('save/13'+'.pkl')




if __name__ == '__main__':
    #parse(test_batch_size=1000)
    #group()
    # toPkl('data/13.txt')
    print(torch.cuda.is_available())






