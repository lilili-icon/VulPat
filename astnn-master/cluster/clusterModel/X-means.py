# coding=utf-8
from sklearn.cluster import KMeans
from sklearn import datasets
import numpy as np
import math
import csv
import pandas as pd
import os


def mykmeans(data, target, k, iteration, ):
    '''
    Parameters
    ----------
    data: array or sparse matrix, shape (n_samples, n_features)
    target: 样本标签，shape(1,n_samples)
    k: cludter number
    iteration: 聚类最大循环次数
    return
    ----------
    result: 聚类结果和集群的失真
    '''
    model = KMeans(n_clusters=k, n_jobs=4, max_iter=iteration)  # 分为k类, 并发数4
    model.fit(data)  # 开始聚类
    # 聚类结果
    labels = model.labels_
    centers = model.cluster_centers_
    distortion = model.inertia_
    clusters = {}
    for i in range(k):
        clusters[i] = {}
        clusters[i]['target'] = []
        clusters[i]['data'] = []
        clusters[i]['cluster_centers'] = centers[i]
    for i in range(len(labels)):
        clusters[labels[i]]['target'].append(target[i])
        clusters[labels[i]]['data'].append(data[i])
    result = {}
    result['clusters'] = clusters
    result['distortion'] = distortion
    return result


def oldbic(n, d, distortion):
    '''
    Parameters
    ----------
    n: 样本总数量
    d: 样本维度
    distortion: 集群失真
    return
    ----------
    L: BIC 分数
    '''
    variance = distortion / (n - 1)
    p1 = -n * math.log(math.pi * 2)
    p2 = -n * d * math.log(variance)
    p3 = -(n - 1)
    L = (p1 + p2 + p3) / 2
    numParameters = d + 1
    return L - 0.5 * numParameters * math.log(n)


def newbic(k, n, d, distortion, clustersSize):
    '''
    Parameters
    ----------
    k: 聚类数量
    n: 样本总数量
    d: 样本维度
    distortion: 集群失真
    clustersSize: 每个聚类的样本数量 shape(1,k)
    return
    ----------
    L: BIC 分数
    '''
    variance = distortion / (n - k);
    L = 0.0;
    for i in range(k):
        L += logLikelihood(k, n, clustersSize[i], d, variance)
    numParameters = k + k * d;
    return L - 0.5 * numParameters * math.log(n);


def logLikelihood(k, n, ni, d, variance):
    '''
    Parameters
    ----------
    k: 聚类数量
    n: 样本总数量
    ni: 属于此聚类的样本数
    d: 样本维度
    variance: 集群的估计方差
    return
    ----------
    loglike: 后验概率估计值
    '''
    p1 = -ni * math.log(math.pi * 2);
    p2 = -ni * d * math.log(variance);
    p3 = -(ni - k);
    p4 = ni * math.log(ni);
    p5 = -ni * math.log(n);
    loglike = (p1 + p2 + p3) / 2 + p4 + p5;
    return loglike;


def myxmeans(data, target, kmin, kmax):
    '''
    Parameters
    ----------
    data: array or sparse matrix, shape (n_samples, n_features)
    target: 样本标签，shape(1,n_samples)
    k: cludter number
    iteration: 聚类最大循环次数
    return
    ----------
    result: 聚类结果和集群的失真
    '''
    d = len(data[0])
    k = kmin
    iteration = 400
    init_clusters = mykmeans(data, target, k, iteration)
    while k < kmax:
        wscc = np.zeros((k, 1))  # 每个集群的失真
        for i in range(k):
            center = init_clusters['clusters'][i]['cluster_centers']
            for tmp_sample in init_clusters['clusters'][i]['data']:
                wscc[i] += np.sqrt(np.sum(np.square(np.array(tmp_sample) - np.array(center))))
        split2cluster = {}
        for i in range(k):
            if len(init_clusters['clusters'][i]['data']) < 2:
                continue
            my2means = mykmeans(init_clusters['clusters'][i]['data'], init_clusters['clusters'][i]['target'], 2,
                                iteration)
            oldbicscore = oldbic(len(init_clusters['clusters'][i]['data']), d, wscc[i])
            newbicscore = newbic(2, len(init_clusters['clusters'][i]['data']), d, my2means['distortion'],
                                 [len(my2means['clusters'][0]['data']), len(my2means['clusters'][1]['data'])])
            if newbicscore > oldbicscore:
                split2cluster[i] = my2means
        for key in split2cluster.keys():
            init_clusters['clusters'][key] = split2cluster[key]['clusters'][0]
            init_clusters['clusters'][k] = split2cluster[key]['clusters'][1]
            k += 1
        if split2cluster == {}:
            break
    return init_clusters

def load_data():

    datas = []
    labels = []
    fileHandler = open("../embedding/CWE-119.csv", "r")
    listOfLines = fileHandler.readlines()
    for list in listOfLines:
        list = list.split(',')
        numbers_list = [float(x) for x in list]
        datas.append(numbers_list[:-1])
        labels.append(int(numbers_list[-1]))

    fileHandler.close()
    return datas, labels

def selectData(path,list):
    csv_result = pd.read_csv(path)
    data = csv_result[csv_result["ID"].isin(list)]
    data.to_csv("13.txt")
    return data


if __name__ == '__main__':


    iris = load_data()
    data,target = load_data()

    xmeans_result = myxmeans(data, target, 10, 20)
    list = xmeans_result['clusters'][13]['target']
    # for key in xmeans_result['clusters'].keys():
    #     print('----------------', key, '------------------')
    #     print(xmeans_result['clusters'][key]['target'], '\n')

    path = '../../../data/CWE-119.txt'
    selectData(path,list)
