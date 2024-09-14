# -*- coding: utf-8 -*-
"""
@author: ZZY
"""
import random
import math
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import os
# import cv2
import numpy as np
# import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import csv
from sklearn import metrics
import pandas as pd


def read_datas(file_dir):
    data = []
    for root, dirs, files in os.walk(file_dir):
        data.append(files)
    return data


def save_datas(file_dir, file_name, file_in):
    for i in range(len(file_name)):
        data = []
        with open(file_dir + file_name[int(i)], 'r')as f:
            for line in f.readlines():
                # data.append(line.strip('\n').strip())
                lines = line.replace('\n', '').strip()
                if not len(lines):
                    continue
                else:
                    data.append(lines)
        file_in.append(data)
    return file_in



def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False



def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def rmse_count(x_in, s_in, number1, number2):
    # 求均方根误差
    sum = 0
    count = 0
    for i in range(number1):
        for j in range(number2):
            # if s_in[i][j] != 0:
                count += 1
                cha = (x_in[i][j] - s_in[i][j])
                sum += pow(cha, 2)
    rmse = math.sqrt(sum / count)
    return rmse


def me_count(x_in, s_in, number1, number2):
    # 求平均误差
    sum = 0
    count = 0
    for i in range(number1):
        for j in range(number2):
            # if s_in[i][j] != 0:
                count += 1
                cha = (x_in[i][j] - s_in[i][j])
                sum += cha
    me = sum / count
    return me


def cc_count(x_in, s_in, number1, number2):
    # 求相关系数
    average_x = np.mean(x_in)
    average_s = np.mean(s_in)
    shang = 0
    xia_1 = 0
    xia_2 = 0
    for i in range(number1):
        for j in range(number2):
            # if s_in[i][j] != 0:
                shang += (x_in[i][j] - average_x) * (s_in[i][j] - average_s)
                xia_1 += pow((x_in[i][j] - average_x), 2)
                xia_2 += pow((s_in[i][j] - average_s), 2)
    cc = shang / (math.sqrt(xia_1) * math.sqrt(xia_2))
    return cc


def bias_count(x_in, s_in, number1, number2):
    # 求相对偏差
    shang = 0
    xia = np.sum(x_in)
    for i in range(number1):
        for j in range(number2):
            # if s_in[i][j] != 0:
                shang += x_in[i][j]-s_in[i][j]
    bias = (shang / xia) * 100
    return bias




if __name__ == '__main__':
    # ==============================================================================

    # ==============================================================================
    a_path = 'F:\\结果\\shirun\\global\\3-5\\Early\\s\\'
    b_path = 'F:\\结果\\shirun\\global\\3-5\\NN\\s\\'
    c_path = 'F:\\结果\\shirun\\global\\3-5\\cpc\\s\\'
    d_path = 'F:\\结果\\shirun\\global\\3-5\\Final\\s\\'

    train = True

    a_names = read_datas(a_path)[0]  ##这里得到的是各个mvk_xx的名字
    b_names = read_datas(b_path)[0]
    c_names = read_datas(c_path)[0]
    d_names = read_datas(d_path)[0]


    a_in = [] #用来存放各个mvk里的数据
    b_in = []
    c_in = []
    d_in = []



    a_in = save_datas(a_path, a_names, a_in)
    # print(a_in)
    b_in = save_datas(b_path,b_names,b_in)
    # print(b_in)
    c_in = save_datas(c_path, c_names, c_in)
    d_in = save_datas(d_path, d_names, d_in)



    a_in = np.array(a_in)
    a_in = a_in.astype('float64')


    b_in = np.array(b_in)
    b_in = b_in.astype('float64')

    c_in = np.array(c_in)
    c_in = c_in.astype('float64')


    d_in = np.array(d_in)
    d_in = d_in.astype('float64')

    number1 = a_in.shape[0]
    number2 = a_in.shape[1]


    CC_early = cc_count(a_in[:, :], c_in[:, :], number1, number2)
    BIAS_early = bias_count(a_in[:, :], c_in[:, :], number1, number2)
    RMSE_early = rmse_count(a_in[:, :], c_in[:, :], number1, number2)
    ME_early = me_count(a_in[:, :], c_in[:, :], number1, number2)

    CC_pre = cc_count(b_in[:,:], c_in[:, :], number1, number2)
    BIAS_pre = bias_count(b_in[:,:], c_in[:, :], number1, number2)
    RMSE_pre = rmse_count(b_in[:,:], c_in[:, :], number1, number2)
    ME_pre = me_count(b_in[:,:], c_in[:, :], number1, number2)

    CC_final = cc_count(d_in[:, :], c_in[:, :], number1, number2)
    BIAS_final = bias_count(d_in[:, :], c_in[:, :], number1, number2)
    RMSE_final = rmse_count(d_in[:, :], c_in[:, :], number1, number2)
    ME_final = me_count(d_in[:, :], c_in[:, :], number1, number2)

    print('#' * 10)
    print("CC_early :", CC_early)
    print("BIAS_early :", BIAS_early)
    print("RMSE_early :", RMSE_early)
    print("ME_early :", ME_early)
    print('*' * 10)
    print("CC_pre:", CC_pre)
    print("BIAS_pre:", BIAS_pre)
    print("RMSE_pre:", RMSE_pre)
    print("ME_pre:", ME_pre)
    print('*' * 10)
    print("CC_final:", CC_final)
    print("BIAS_final:", BIAS_final)
    print("RMSE_final:", RMSE_final)
    print("ME_final:", ME_final)

