# -*- coding: utf-8 -*-

import re
import csv
from more_itertools import chunked
import pandas as pd


# 首先是整理了一下吧所有的换行、注释、unisit替换、标志去除 ""
def format_string(string):
    for char in  ['uint32_t','uint16_t','uint8_t','uint64_t','bool','int8_t','int16_t','int32_t','int64_t']:
        string = string.replace(char, 'int')
    string = string.replace('""','"').replace('\n','=====')
    string = re.sub(r'/\*[\w\W]*?\*/|^//.*?$|^//.*?$', ' ', string)
    string = re.sub(r',=====					__location__[\w\W]*?\%d[\w\W]*? \%s[\w\W]*?====="', ',__location__ %d %s',string)
    string = re.sub(r'__location__[\w\W]*?\%s[\w\W]*?====="', ' __location__ %s', string)
    string = string.replace('=====','')
    return string

def preprocess_test_data(test_batch_size=1000):
    csv.field_size_limit(500 * 1024 * 1024)
    with open('allData.csv',encoding='UTF-8') as f :
        data = csv.reader(f)
        batched_data = chunked(data, test_batch_size)
        print("start processing")
        for batch_data in batched_data:
            for CVE_ID,CWE_D,func_before in batch_data:
                func_before =format_string(func_before)
                chunk_source_list = [[CVE_ID,CWE_D,func_before]]
                df = pd.DataFrame(data=chunk_source_list)
                file = "allData_2.csv"
                df.to_csv(file, header=False, index=False, mode='a')

if __name__ == '__main__':
    preprocess_test_data(test_batch_size=1000)

