# -*- coding: utf-8 -*-

import math
import numpy
import ast
import os
import pandas as pd

Similarity = []


def point(x, y):
    return '[' + str(x) + ',' + str(y) + ']'


class CodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.seq = []

    def generic_visit(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        self.seq.append(type(node).__name__)

    def visit_FunctionDef(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        self.seq.append(type(node).__name__)

    def visit_Assign(self, node):
        self.seq.append(type(node).__name__)


class CodeParse(object):
    def __init__(self, fileA, fileB):
        self.visitorB = None
        self.visitorA = None
        self.codeA = open(fileA, encoding="utf-8").read()
        self.codeB = open(fileB, encoding="utf-8").read()
        self.nodeA = ast.parse(self.codeA)
        self.nodeB = ast.parse(self.codeB)
        self.seqA = ""
        self.seqB = ""
        self.work()

    def work(self):
        self.visitorA = CodeVisitor()
        self.visitorA.visit(self.nodeA)
        self.seqA = self.visitorA.seq
        self.visitorB = CodeVisitor()
        self.visitorB.visit(self.nodeB)
        self.seqB = self.visitorB.seq


class CalculateSimilarity(object):
    def __init__(self, A, B, W, M, N):
        self.A = A
        self.B = B
        self.W = W
        self.M = M
        self.N = N
        self.similarity = []
        self.string1 = []
        self.string2 = []
        self.SimthWaterman(self.A, self.B, self.W)

    def score(self, a, b):
        if a == b:
            return self.M
        else:
            return self.N

    def traceback(self, A, B, H, path, value, result):
        if value:
            temp = value[0]
            result.append(temp)
            value = path[temp]
            x = int((temp.split(',')[0]).strip('['))
            y = int((temp.split(',')[1]).strip(']'))
        else:
            return
        if H[x, y] == 0:  # 终止条件
            xx = 0
            yy = 0
            sim = 0
            s1 = ''
            s2 = ''
            md = ''
            for item in range(len(result) - 2, -1, -1):
                position = result[item]
                x = int((position.split(',')[0]).strip('['))
                y = int((position.split(',')[1]).strip(']'))
                if x == xx:
                    s1 += '-'
                    s2 += B[y - 1]
                    md += ' '
                    pass
                elif y == yy:
                    s1 += A[x - 1]
                    s2 += '-'
                    md += ' '
                    pass
                else:
                    sim = sim + 1
                    s1 += A[x - 1]
                    s2 += B[y - 1]
                    md += '|'
                xx = x
                yy = y
            self.similarity.append(sim * 2 / (len(A) + len(B)))
            # 输出最佳匹配序列
            # print('s1: %s' % s1)
            # print('    ' + md)
            # print('s2: %s' % s2)
            self.string1.append(s1)
            self.string2.append(s2)

        else:
            self.traceback(A, B, H, path, value, result)

    def SimthWaterman(self, A, B, W):
        n, m = len(A), len(B)
        H = numpy.zeros([n + 1, m + 1], int)
        path = {}
        for i in range(0, n + 1):
            for j in range(0, m + 1):
                if i == 0 or j == 0:
                    path[point(i, j)] = []
                else:
                    s = self.score(A[i - 1], B[j - 1])
                    L = H[i - 1, j - 1] + s
                    P = H[i - 1, j] - W
                    Q = H[i, j - 1] - W
                    H[i, j] = max(L, P, Q, 0)

                    # 添加进路径
                    path[point(i, j)] = []
                    if math.floor(L) == H[i, j]:
                        path[point(i, j)].append(point(i - 1, j - 1))
                    if math.floor(P) == H[i, j]:
                        path[point(i, j)].append(point(i - 1, j))
                    if math.floor(Q) == H[i, j]:
                        path[point(i, j)].append(point(i, j - 1))

        end = numpy.argwhere(H == numpy.max(H))
        for pos in end:
            key = point(pos[0], pos[1])
            value = path[key]
            result = [key]
            self.traceback(A, B, H, path, value, result)

    def Answer(self):  # 取均值
        return sum(self.similarity) / len(self.similarity)

    def SS(self):  # 取均值
        return self.string1,self.string2



def main(A,B):
    #input_file1 = '../data/CWE-119.pkl'
    #blocks = pd.read_pickle(input_file1)
    #generate_block_seqs(input_file1)
    # dictionary_and_embedding(imput_file1,imput_file,128)

    RES = CalculateSimilarity(A, B, 1, 1, -1 / 3)
    # print(RES.Answer())
    return RES.Answer()


def pairCom():

    return

def bubbleSort_Recursion(arr, size):
    if (size < 2):
        return
    for i in range(1, size):
        if (arr[i - 1] > arr[i]):
            temp = arr[i]
            arr[i] = arr[i - 1]
            arr[i - 1] = temp
    bubbleSort_Recursion(arr, size - 1)

def str2list(sting):
     str = sting.strip('[').strip("]").strip('"').split('\,')

     return str



if __name__ == "__main__":


    # score = main(s1, s2)
    # print(score)
    items = pd.read_csv("result/test.tsv")
    test = ''
    entry = ''
    for count,item in items.iterrows():
        test = item[5]
        flag = True
        maxscore = 0
        files = os.listdir("result/patterns/")
        for file in files:
            entrys = pd.read_csv("result/patterns/"+file)
            print("正在比较"+file+"文件")

            for _,entry in entrys.iterrows():
                target = entry[4]
                try:
                    score = main(test, target)
                    data = [item[0], item[4], entry[3], entry[0], file, score]
                    df = pd.DataFrame(data=[data])
                    df.to_csv(file+".csv", header=False, index=False, mode='a')
                    if maxscore < score:
                        maxscore = score
                        data = [item[0], item[4], entry[3], entry[0], file, maxscore]
                        df = pd.DataFrame(data=[data])
                        if flag :
                            df.to_csv("output.csv", header=False, index=False, mode='a')
                            flag = False
                        else:
                            df.to_csv("output.csv", header=False, index=False)
                    else:
                        continue
                except Exception as e:
                    data = [item[0], entry[0],item[4], entry[3],file]
                    df = pd.DataFrame(data=[data])
                    df.to_csv("output_error.csv", header=False, index=False, mode='a')
            print(file + "文件比较完成")
        print("完成test"+count)









