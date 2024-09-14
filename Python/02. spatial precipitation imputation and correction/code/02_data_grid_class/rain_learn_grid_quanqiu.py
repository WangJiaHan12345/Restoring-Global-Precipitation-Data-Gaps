###########################################################################################
# 只出结果
# -*- coding: utf-8 -*-
"""
@author: ZZY
"""
import random
import math
# import tensorflow as tf
# import tensorflow.compat.v1 as tf
import tensorflow._api.v2.compat.v1 as tf

# tf.enable_eager_execution()
tf.disable_v2_behavior()
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


def rmse_count(x_in, s_in,number1,number2):
    # 求均方根误差
    sum = 0
    count = 0
    for k in range(number2):
        count += 1
        cha = (x_in[k] - s_in[k])
        sum += pow(cha, 2)
    rmse = math.sqrt(sum / count)
    return rmse


def mae_count(x_in, s_in,number1, number2):
    # 求平均误差
    sum = 0
    count = 0
    for k in range(number2):
            count += 1
            cha = abs(x_in[k] - s_in[k])
            sum += cha
    me = sum / count
    return me


def cc_count(x_in, s_in, number1,number2):
    # 求相关系数
    average_x = np.mean(x_in)
    average_s = np.mean(s_in)
    shang = 0
    xia_1 = 0
    xia_2 = 0

    for k in range(number2):
        shang += (x_in[k] - average_x) * (s_in[k] - average_s)
        xia_1 += pow((x_in[k] - average_x), 2)
        xia_2 += pow((s_in[k] - average_s), 2)
    if xia_1 == 0:
        xia_1 = 3.6
    if xia_2 == 0:
        xia_2 = 3.6
    cc = shang / (math.sqrt(xia_1) * math.sqrt(xia_2))
    return cc


def bias_count(x_in, s_in,number1, number2):
    # 求相对偏差
    shang = 0
    xia = np.sum(s_in)
    if xia == 0:
        xia = 0.1
    for k in range(number2):
            shang += x_in[k]-s_in[k]
    bias = (shang / xia) * 100
    return bias

def abias_count(x_in, s_in, number1,number2):
    # 求绝对偏差
    shang = 0
    xia = np.sum(s_in)
    if xia == 0:
        xia = 0.1
    for k in range(number2):
          shang += abs(x_in[k]-s_in[k])
    abias = (shang / xia) * 100
    return abias


def pod(x_in, s_in,number1, number2):
    # 求pod
    n10 = 0
    n01 = 0
    n11 = 0
    for k in range(number2):
        # CPC
        if s_in[k] >= 2.4 and x_in[k] < 2.4:
            n01 = n01 + 1
        # 卫星
        if x_in[k] >= 2.4 and s_in[k] < 2.4:
            n10 = n10 + 1
        #     cpc+卫星
        if x_in[k] >= 2.4 and s_in[k] >= 2.4:
            n11 = n11 +1
    if n11==0 and n01==0:
        pod=1
    else:
         pod = n11/(n11+n01)
    return pod

def far(x_in, s_in,number1, number2):
    # 求pod
    n10 = 0
    n01 = 0
    n11 = 0
    for k in range(number2):
        # CPC
        if s_in[k] >= 2.4 and x_in[k] < 2.4:
            n01 = n01 + 1
        # 卫星
        if x_in[k] >= 2.4 and s_in[k] < 2.4:
            n10 = n10 + 1
        #     cpc+卫星
        if x_in[k] >= 2.4 and s_in[k] >= 2.4:
            n11 = n11 +1
    if n11==0 and n10==0:
        far=0
    else:
        far = n10/(n11+n10)
    return far

def csi(x_in, s_in,number1, number2):
    # 求pod
    n10 = 0
    n01 = 0
    n11 = 0
    for k in range(number2):
        # CPC
        if s_in[k] >= 2.4 and x_in[k] < 2.4:
            n01 = n01 + 1
        # 卫星
        if x_in[k] >= 2.4 and s_in[k] < 2.4:
            n10 = n10 + 1
        #     cpc+卫星
        if x_in[k] >= 2.4 and s_in[k] >= 2.4:
            n11 = n11 +1
    if n11==0 and n01==0 and n10==0:
        csi=1
    else:
        csi = n11/(n11+n10+n01)
    return csi


if __name__ == '__main__':

    a_path = 'G:\\青藏高原\\时间\\季度\\只包含站点的网格(rnt)\\综合\\rnt\\'
    b_path = 'G:\\青藏高原\\时间\\季度\\只包含站点的网格(rnt)\\综合\\ANN\\'
    c_path = 'G:\\青藏高原\\时间\\季度\\只包含站点的网格(rnt)\\综合\\国家气象局\\'
    # d_path = 'G:\\全球\\时间预测结果\\地面站点_Early\\ganhan\\总\\1-12\\final\\'

    a_names = read_datas(a_path)[0]
    b_names = read_datas(b_path)[0]
    c_names = read_datas(c_path)[0]
    # d_names = read_datas(d_path)[0]


    a_in = []  # 用来存放各个mvk里的数据
    b_in = []
    c_in = []
    # d_in = []


    a_in = save_datas(a_path, a_names, a_in)
    b_in = save_datas(b_path, b_names, b_in)
    c_in = save_datas(c_path, c_names, c_in)
    # d_in = save_datas(d_path, d_names, d_in)


    a_in = np.array(a_in)
    a_in = a_in.astype('float64')

    b_in = np.array(b_in)
    b_in = b_in.astype('float64')

    c_in = np.array(c_in)
    c_in = c_in.astype('float64')

    # d_in = np.array(d_in)
    # d_in = d_in.astype('float64')



    number1 = c_in.shape[0]
    number2 = c_in.shape[1]
    print(number1,number2)

    count = 168  # 总网格
    result_mvk_cc = np.zeros((count, 1))
    result_mvk_bias = np.zeros((count, 1))
    result_mvk_abias = np.zeros((count, 1))
    result_mvk_rmse = np.zeros((count, 1))
    result_mvk_mae = np.zeros((count, 1))
    result_mvk_pod = np.zeros((count, 1))
    result_mvk_far = np.zeros((count, 1))
    result_mvk_csi = np.zeros((count, 1))

    result_gauge_cc = np.zeros((count, 1))
    result_gauge_bias = np.zeros((count, 1))
    result_gauge_abias = np.zeros((count, 1))
    result_gauge_rmse = np.zeros((count, 1))
    result_gauge_mae = np.zeros((count, 1))
    result_gauge_pod = np.zeros((count, 1))
    result_gauge_far = np.zeros((count, 1))
    result_gauge_csi = np.zeros((count, 1))

    result_ann_cc = np.zeros((count, 1))
    result_ann_bias = np.zeros((count, 1))
    result_ann_abias = np.zeros((count, 1))
    result_ann_rmse = np.zeros((count, 1))
    result_ann_mae = np.zeros((count, 1))
    result_ann_pod = np.zeros((count, 1))
    result_ann_far = np.zeros((count, 1))
    result_ann_csi = np.zeros((count, 1))


    for i in range(number1):
        CC_mvk = cc_count(a_in[i, :], c_in[i, :],i, number2)
        BIAS_mvk = bias_count(a_in[i, :], c_in[i, :],i,  number2)
        ABIAS_mvk = abias_count(a_in[i, :], c_in[i, :],i,  number2)
        RMSE_mvk = rmse_count(a_in[i, :], c_in[i, :],i,  number2)
        MAE_mvk = mae_count(a_in[i, :], c_in[i, :],i,  number2)
        POD_mvk = pod(a_in[i, :], c_in[i, :],i,  number2)
        FAR_mvk = far(a_in[i, :], c_in[i, :],i, number2)
        CSI_mvk = csi(a_in[i, :], c_in[i, :],i,  number2)

        result_mvk_cc[i, 0] = CC_mvk
        result_mvk_bias[i, 0] = BIAS_mvk
        result_mvk_abias[i, 0] = ABIAS_mvk
        result_mvk_rmse[i, 0] = RMSE_mvk
        result_mvk_mae[i, 0] = MAE_mvk
        result_mvk_pod[i, 0] = POD_mvk
        result_mvk_far[i, 0] = FAR_mvk
        result_mvk_csi[i, 0] = CSI_mvk
        #
        CC_pre = cc_count(b_in[i, :], c_in[i, :],i, number2)
        BIAS_pre = bias_count(b_in[i, :], c_in[i, :],i, number2)
        ABIAS_pre = abias_count(b_in[i, :], c_in[i, :],i, number2)
        RMSE_pre = rmse_count(b_in[i, :], c_in[i, :],i, number2)
        MAE_pre = mae_count(b_in[i, :], c_in[i, :],i, number2)
        POD_pre = pod(b_in[i, :], c_in[i, :],i, number2)
        FAR_pre = far(b_in[i, :], c_in[i, :],i, number2)
        CSI_pre = csi(b_in[i, :], c_in[i, :],i, number2)

        result_ann_cc[i, 0] = CC_pre
        result_ann_bias[i, 0] = BIAS_pre
        result_ann_abias[i, 0] = ABIAS_pre
        result_ann_rmse[i, 0] = RMSE_pre
        result_ann_mae[i, 0] = MAE_pre
        result_ann_pod[i, 0] = POD_pre
        result_ann_far[i, 0] = FAR_pre
        result_ann_csi[i, 0] = CSI_pre
        #
        # CC_gauge = cc_count(d_in[i, :], c_in[i, :],i, number2)
        # BIAS_gauge = bias_count(d_in[i, :], c_in[i, :],i, number2)
        # ABIAS_gauge = abias_count(d_in[i, :], c_in[i, :],i, number2)
        # RMSE_gauge = rmse_count(d_in[i, :], c_in[i, :],i, number2)
        # MAE_gauge = mae_count(d_in[i, :], c_in[i, :],i, number2)
        # POD_gauge = pod(d_in[i, :], c_in[i, :],i, number2)
        # FAR_gauge = far(d_in[i, :], c_in[i, :],i, number2)
        # CSI_gauge = csi(d_in[i, :], c_in[i, :],i, number2)
        #
        # result_gauge_cc[i, 0] = CC_gauge
        # result_gauge_bias[i, 0] = BIAS_gauge
        # result_gauge_abias[i, 0] = ABIAS_gauge
        # result_gauge_rmse[i, 0] = RMSE_gauge
        # result_gauge_mae[i, 0] = MAE_gauge
        # result_gauge_pod[i, 0] = POD_gauge
        # result_gauge_far[i, 0] = FAR_gauge
        # result_gauge_csi[i, 0] = CSI_gauge

    path_save = 'G:\\青藏高原\\时间\\季度\\只包含站点的网格(rnt)\\综合\\指标\\'
    mkdir(path_save)  # 创建一个目录存放结果

    a = path_save + 'result_early_cc.txt'
    with open(a, 'w')as f:
        for i in range(len(result_mvk_cc)):
            for j in range(1):
                f.writelines(str(result_mvk_cc[i, j]))
                f.writelines('\n')

    a1 = path_save + 'result_early_bias.txt'
    with open(a1, 'w')as f:
        for i in range(len(result_mvk_bias)):
            for j in range(1):
                f.writelines(str(result_mvk_bias[i, j]))
                f.writelines('\n')

    a11 = path_save + 'result_early_abias.txt'
    with open(a11, 'w')as f:
        for i in range(len(result_mvk_abias)):
            for j in range(1):
                f.writelines(str(result_mvk_abias[i, j]))
                f.writelines('\n')

    a2 = path_save + 'result_early_rmse.txt'
    with open(a2, 'w')as f:
        for i in range(len(result_mvk_rmse)):
            for j in range(1):
                f.writelines(str(result_mvk_rmse[i, j]))
                f.writelines('\n')

    a3 = path_save + 'result_early_mae.txt'
    with open(a3, 'w')as f:
        for i in range(len(result_mvk_mae)):
            for j in range(1):
                f.writelines(str(result_mvk_mae[i, j]))
                f.writelines('\n')

    a4 = path_save + 'result_early_pod.txt'
    with open(a4, 'w')as f:
        for i in range(len(result_mvk_pod)):
            for j in range(1):
                f.writelines(str(result_mvk_pod[i, j]))
                f.writelines('\n')

    a5 = path_save + 'result_early_far.txt'
    with open(a5, 'w')as f:
        for i in range(len(result_mvk_far)):
            for j in range(1):
                f.writelines(str(result_mvk_far[i, j]))
                f.writelines('\n')

    a6 = path_save + 'result_early_csi.txt'
    with open(a6, 'w')as f:
        for i in range(len(result_mvk_csi)):
            for j in range(1):
                f.writelines(str(result_mvk_csi[i, j]))
                f.writelines('\n')

    # b = path_save + 'result_final_cc.txt'
    # with open(b, 'w')as f:
    #     for i in range(len(result_gauge_cc)):
    #         for j in range(1):
    #             f.writelines(str(result_gauge_cc[i, j]))
    #             f.writelines('\n')
    #
    # b1 = path_save + 'result_final_bias.txt'
    # with open(b1, 'w')as f:
    #     for i in range(len(result_gauge_bias)):
    #         for j in range(1):
    #             f.writelines(str(result_gauge_bias[i, j]))
    #             f.writelines('\n')
    #
    # b11 = path_save + 'result_final_abias.txt'
    # with open(b11, 'w')as f:
    #     for i in range(len(result_gauge_abias)):
    #         for j in range(1):
    #             f.writelines(str(result_gauge_abias[i, j]))
    #             f.writelines('\n')
    #
    # b2 = path_save + 'result_final_rmse.txt'
    # with open(b2, 'w')as f:
    #     for i in range(len(result_gauge_rmse)):
    #         for j in range(1):
    #             f.writelines(str(result_gauge_rmse[i, j]))
    #             f.writelines('\n')
    #
    # b3 = path_save + 'result_final_mae.txt'
    # with open(b3, 'w')as f:
    #     for i in range(len(result_gauge_mae)):
    #         for j in range(1):
    #             f.writelines(str(result_gauge_mae[i, j]))
    #             f.writelines('\n')
    #
    # b4 = path_save + 'result_final_pod.txt'
    # with open(b4, 'w')as f:
    #     for i in range(len(result_gauge_pod)):
    #         for j in range(1):
    #             f.writelines(str(result_gauge_pod[i, j]))
    #             f.writelines('\n')
    #
    # b5 = path_save + 'result_final_far.txt'
    # with open(b5, 'w')as f:
    #     for i in range(len(result_gauge_far)):
    #         for j in range(1):
    #             f.writelines(str(result_gauge_far[i, j]))
    #             f.writelines('\n')
    #
    # b6 = path_save + 'result_final_csi.txt'
    # with open(b6, 'w')as f:
    #     for i in range(len(result_gauge_csi)):
    #         for j in range(1):
    #             f.writelines(str(result_gauge_csi[i, j]))
    #             f.writelines('\n')


    c = path_save + 'result_ann_cc.txt'
    with open(c, 'w')as f:
        for i in range(len(result_ann_cc)):
            for j in range(1):
                f.writelines(str(result_ann_cc[i, j]))
                f.writelines('\n')

    c1 = path_save + 'result_ann_bias.txt'
    with open(c1, 'w')as f:
        for i in range(len(result_ann_bias)):
            for j in range(1):
                f.writelines(str(result_ann_bias[i, j]))
                f.writelines('\n')

    c11 = path_save + 'result_ann_abias.txt'
    with open(c11, 'w')as f:
        for i in range(len(result_ann_abias)):
            for j in range(1):
                f.writelines(str(result_ann_abias[i, j]))
                f.writelines('\n')

    c2 = path_save + 'result_ann_rmse.txt'
    with open(c2, 'w')as f:
        for i in range(len(result_ann_rmse)):
            for j in range(1):
                f.writelines(str(result_ann_rmse[i, j]))
                f.writelines('\n')

    c3 = path_save + 'result_ann_mae.txt'
    with open(c3, 'w')as f:
        for i in range(len(result_ann_mae)):
            for j in range(1):
                f.writelines(str(result_ann_mae[i, j]))
                f.writelines('\n')

    c4 = path_save + 'result_ann_pod.txt'
    with open(c4, 'w')as f:
        for i in range(len(result_ann_pod)):
            for j in range(1):
                f.writelines(str(result_ann_pod[i, j]))
                f.writelines('\n')

    c5 = path_save + 'result_ann_far.txt'
    with open(c5, 'w')as f:
        for i in range(len(result_ann_far)):
            for j in range(1):
                f.writelines(str(result_ann_far[i, j]))
                f.writelines('\n')

    c6 = path_save + 'result_ann_csi.txt'
    with open(c6, 'w')as f:
        for i in range(len(result_ann_csi)):
            for j in range(1):
                f.writelines(str(result_ann_csi[i, j]))
                f.writelines('\n')

