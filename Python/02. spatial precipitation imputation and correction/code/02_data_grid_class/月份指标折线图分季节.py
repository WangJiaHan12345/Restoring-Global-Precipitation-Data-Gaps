###########################################################################################
#只出结果
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


def SD_count(x_in, s_in, number1, number2):
    # 求标准差
    sum = 0
    count = 0
    average_s = np.mean(s_in)
    for j in range(number1):
        for k in range(number2):
            count += 1
            cha = (x_in[j][k] - average_s)
            sum += pow(cha, 2)
    sd = math.sqrt(sum / (count-1))
    return sd


def rmse_count(x_in, s_in, number1, number2):
    # 求均方根误差
    sum = 0
    count = 0
    for i in range(number1):
        for j in range(number2):
            count += 1
            cha = (x_in[i][j] - s_in[i][j])
            sum += pow(cha, 2)
    rmse = math.sqrt(sum / count)
    return rmse


def cc_count(x_in, s_in, number1, number2):
    # 求相关系数
    average_x = np.mean(x_in)
    average_s = np.mean(s_in)
    shang = 0
    xia_1 = 0
    xia_2 = 0
    for i in range(number1):
        for j in range(number2):
            shang += (x_in[i][j] - average_x) * (s_in[i][j] - average_s)
            xia_1 += pow((x_in[i][j] - average_x), 2)
            xia_2 += pow((s_in[i][j] - average_s), 2)
    cc = shang / (math.sqrt(xia_1) * math.sqrt(xia_2))
    return cc


def mae_count(x_in, s_in, number1, number2):
    sum = 0
    count = 0
    for i in range(number1):
        for j in range(number2):
            count += 1
            cha = abs(x_in[i][j] - s_in[i][j])
            sum += cha
    mae = sum / count
    return mae



def bias_count(x_in, s_in, number1, number2):
    # 求相对偏差
    shang = 0
    xia = np.sum(s_in)
    for i in range(number1):
        for j in range(number2):
            shang += x_in[i][j] - s_in[i][j]
    bias = (shang / xia) * 100
    return bias


def abias_count(x_in, s_in, number1, number2):
    # 求绝对偏差
    shang = 0
    xia = np.sum(s_in)
    for j in range(number1):
        for k in range(number2):
            shang += abs(x_in[j][k]-s_in[j][k])
    abias = (shang / xia) * 100
    return abias


def pod(x_in, s_in, number1, number2):
    # 求pod
    n10 = 0
    n01 = 0
    n11 = 0
    for j in range(number1):
        for k in range(number2):
        # CPC
            if s_in[j][k] >= 2.4 and x_in[j][k] < 2.4:
                n01 = n01 + 1
            # 卫星
            if x_in[j][k] >= 2.4 and s_in[j][k] < 2.4:
                n10 = n10 + 1
            #     cpc+卫星
            if x_in[j][k] >= 2.4 and s_in[j][k] >= 2.4:
                n11 = n11 +1

    pod = n11/(n11+n01)
    return pod

def far(x_in, s_in, number1, number2):
    # 求pod
    n10 = 0
    n01 = 0
    n11 = 0
    for j in range(number1):
        for k in range(number2):
            # CPC
            if s_in[j][k] >= 2.4 and x_in[j][k] < 2.4:
                n01 = n01 + 1
            # 卫星
            if x_in[j][k] >= 2.4 and s_in[j][k] < 2.4:
                n10 = n10 + 1
            #     cpc+卫星
            if x_in[j][k] >= 2.4 and s_in[j][k] >= 2.4:
                n11 = n11 + 1
    far = n10/(n11+n10)
    return far

def csi(x_in, s_in, number1, number2):
    # 求pod
    n10 = 0
    n01 = 0
    n11 = 0
    for j in range(number1):
        for k in range(number2):
            # CPC
            if s_in[j][k] >= 2.4 and x_in[j][k] < 2.4:
                n01 = n01 + 1
            # 卫星
            if x_in[j][k] >= 2.4 and s_in[j][k] < 2.4:
                n10 = n10 + 1
            #     cpc+卫星
            if x_in[j][k] >= 2.4 and s_in[j][k] >= 2.4:
                n11 = n11 + 1

    csi = n11/(n11+n10+n01)
    return csi

# banqius = ['shirun','banshirun','banganhan','ganhan']
# for banqiu in banqius:
# earths = ['global_n','global_s']
# for earth in earths:
seasons = ['3-5','6-8','9-11','12-2']
for season in seasons:
    if __name__ == '__main__':
        # a_path = 'F:\\全球\\时间预测结果\\global_ns\\'+ earth +'\\'+ season +'\\Early\\'
        # b_path = 'F:\\全球\\时间预测结果\\global_ns\\'+ earth +'\\'+ season +'\\ANN\\'
        # c_path = 'F:\\全球\\时间预测结果\\global_ns\\'+ earth +'\\'+ season +'\\cpc\\'
        # d_path = 'F:\\全球\\时间预测结果\\global_ns\\'+ earth +'\\'+ season +'\\Final\\'
        # e_path = 'F:\\全球\\时间预测结果\\global_ns\\'+ earth +'\\'+ season +'\\DNN\\'

        a_path = 'F:\\青藏高原\\空间\\季节\\'+ season +'\\gsmap_mvk\\'
        b_path = 'F:\\青藏高原\\空间\\季节\\'+ season +'\\ANN\\'
        c_path = 'F:\\青藏高原\\空间\\季节\\'+ season +'\\国家气象局\\'
        d_path = 'F:\\青藏高原\\空间\\季节\\'+ season +'\\gsmap_gauge\\'


        a_names = read_datas(a_path)[0]
        b_names = read_datas(b_path)[0]
        c_names = read_datas(c_path)[0]
        d_names = read_datas(d_path)[0]
        # e_names = read_datas(e_path)[0]
        # f_names = read_datas(f_path)[0]


        a_in = [] #用来存放各个mvk里的数据
        b_in = []
        c_in = []
        d_in = []
        # e_in = []
        # f_in = []


        a_in = save_datas(a_path, a_names, a_in)
        b_in = save_datas(b_path,b_names,b_in)
        c_in = save_datas(c_path, c_names, c_in)
        d_in = save_datas(d_path,d_names,d_in)
        # e_in = save_datas(e_path, e_names, e_in)
        # f_in = save_datas(f_path, f_names, f_in)



        a_in = np.array(a_in)
        a_in = a_in.astype('float64')

        b_in = np.array(b_in)
        b_in = b_in.astype('float64')

        c_in = np.array(c_in)
        c_in = c_in.astype('float64')

        d_in = np.array(d_in)
        d_in = d_in.astype('float64')

        # e_in = np.array(e_in)
        # e_in = e_in.astype('float64')


        # result_early_cc=np.zeros((12,1))
        # result_early_bias = np.zeros((12, 1))
        # result_early_rmse = np.zeros((12, 1))
        # result_early_mae = np.zeros((12, 1))
        # result_early_abias = np.zeros((12, 1))
        # result_early_pod = np.zeros((12, 1))
        # result_early_far = np.zeros((12, 1))
        # result_early_csi = np.zeros((12, 1))
        #
        #
        # result_final_cc = np.zeros((12, 1))
        # result_final_bias = np.zeros((12, 1))
        # result_final_rmse = np.zeros((12, 1))
        # result_final_mae = np.zeros((12, 1))
        # result_final_abias = np.zeros((12, 1))
        # result_final_pod = np.zeros((12, 1))
        # result_final_far = np.zeros((12, 1))
        # result_final_csi = np.zeros((12, 1))
        #
        #
        #
        # result_ann_cc = np.zeros((12, 1))
        # result_ann_bias = np.zeros((12, 1))
        # result_ann_rmse = np.zeros((12, 1))
        # result_ann_mae = np.zeros((12, 1))
        # result_ann_abias = np.zeros((12, 1))
        # result_ann_pod = np.zeros((12, 1))
        # result_ann_far = np.zeros((12, 1))
        # result_ann_csi = np.zeros((12, 1))
        #
        #
        # result_nn_cc = np.zeros((12, 1))
        # result_nn_bias = np.zeros((12, 1))
        # result_nn_rmse = np.zeros((12, 1))
        # result_nn_mae = np.zeros((12, 1))
        # result_nn_abias = np.zeros((12, 1))
        # result_nn_pod = np.zeros((12, 1))
        # result_nn_far = np.zeros((12, 1))
        # result_nn_csi = np.zeros((12, 1))

        # result_early_cc = np.zeros((3, 1))
        # result_early_bias = np.zeros((3, 1))
        # result_early_rmse = np.zeros((3, 1))
        # result_early_mae = np.zeros((3, 1))
        # result_early_abias = np.zeros((3, 1))
        # result_early_pod = np.zeros((3, 1))
        # result_early_far = np.zeros((3, 1))
        # result_early_csi = np.zeros((3, 1))
        #
        # result_final_cc = np.zeros((3, 1))
        # result_final_bias = np.zeros((3, 1))
        # result_final_rmse = np.zeros((3, 1))
        # result_final_mae = np.zeros((3, 1))
        # result_final_abias = np.zeros((3, 1))
        # result_final_pod = np.zeros((3, 1))
        # result_final_far = np.zeros((3, 1))
        # result_final_csi = np.zeros((3, 1))
        #
        # result_ann_cc = np.zeros((3, 1))
        # result_ann_bias = np.zeros((3, 1))
        # result_ann_rmse = np.zeros((3, 1))
        # result_ann_mae = np.zeros((3, 1))
        # result_ann_abias = np.zeros((3, 1))
        # result_ann_pod = np.zeros((3, 1))
        # result_ann_far = np.zeros((3, 1))
        # result_ann_csi = np.zeros((3, 1))

        # result_nn_cc = np.zeros((3, 1))
        # result_nn_bias = np.zeros((3, 1))
        # result_nn_rmse = np.zeros((3, 1))
        # result_nn_mae = np.zeros((3, 1))
        # result_nn_abias = np.zeros((3, 1))
        # result_nn_pod = np.zeros((3, 1))
        # result_nn_far = np.zeros((3, 1))
        # result_nn_csi = np.zeros((3, 1))

        # 青藏高原部分
        result_early_cc = np.zeros((6, 1))
        result_early_bias = np.zeros((6, 1))
        result_early_rmse = np.zeros((6, 1))
        result_early_mae = np.zeros((6, 1))
        result_early_abias = np.zeros((6, 1))
        result_early_pod = np.zeros((6, 1))
        result_early_far = np.zeros((6, 1))
        result_early_csi = np.zeros((6, 1))

        result_final_cc = np.zeros((6, 1))
        result_final_bias = np.zeros((6, 1))
        result_final_rmse = np.zeros((6, 1))
        result_final_mae = np.zeros((6, 1))
        result_final_abias = np.zeros((6, 1))
        result_final_pod = np.zeros((6, 1))
        result_final_far = np.zeros((6, 1))
        result_final_csi = np.zeros((6, 1))

        result_ann_cc = np.zeros((6, 1))
        result_ann_bias = np.zeros((6, 1))
        result_ann_rmse = np.zeros((6, 1))
        result_ann_mae = np.zeros((6, 1))
        result_ann_abias = np.zeros((6, 1))
        result_ann_pod = np.zeros((6, 1))
        result_ann_far = np.zeros((6, 1))
        result_ann_csi = np.zeros((6, 1))

        number1 = a_in.shape[0]
        # number2 = a_in.shape[1]

        if season == '3-5':
            # step = [0, 31, 61, 92, 123, 153, 184, 215, 245, 276, 307, 337, 368]  # 3-5
            # step = [0,31,61,92]
            #青藏高原部分
            step = [0,31,61,92,123,153,184]
        elif season == '6-8':
            # step = [0, 30, 61, 92, 122, 153, 184, 214, 245, 276, 306, 337, 368]  # 6-8
            # step = [0,31,61,92]
            # 青藏高原部分
            step = [0, 30, 61, 92, 122, 153, 184]
        elif season == '9-11':
            # step = [0, 30, 61, 91, 121, 152, 182, 212, 243, 273, 303, 334, 364]  # 9-11
            # step = [0,30,61,90]
            # 青藏高原部分
            step = [0, 30, 61, 91, 121, 152, 182]
        else:
            # step=[0,31,59,90,121,150,181,212,240,271,302,330,361]  #12-2
            # step = [0,31,58,87]
            # 青藏高原部分
            step = [0,31,59,90,121,150,181]


        # for i in range(1,13):
        # for i in range(1, 4):
        for i in range(1, 7):

            number2 = a_in[:,step[i-1]:step[i]].shape[1]


            CC_early = cc_count(a_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]], number1, number2)
            BIAS_early = bias_count(a_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]],  number1, number2)
            RMSE_early = rmse_count(a_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]],  number1, number2)
            MAE_early = mae_count(a_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]],   number1, number2)
            ABIAS_early = abias_count(a_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]],   number1, number2)
            POD_early = pod(a_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]],   number1, number2)
            FAR_early = far(a_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]],   number1, number2)
            CSI_early = csi(a_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]],   number1, number2)

            result_early_cc[[i-1], [0]] = CC_early
            result_early_bias[[i-1], [0]] = BIAS_early
            result_early_rmse[[i-1], [0]] = RMSE_early
            result_early_mae[[i-1], [0]] = MAE_early
            result_early_abias[[i-1], [0]] = ABIAS_early
            result_early_pod[[i-1], [0]] = POD_early
            result_early_far[[i-1], [0]] = FAR_early
            result_early_csi[[i-1], [0]] = CSI_early


            CC_ann = cc_count(b_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]],  number1, number2)
            BIAS_ann = bias_count(b_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]],  number1, number2)
            RMSE_ann = rmse_count(b_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]], number1, number2)
            MAE_ann = mae_count(b_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]],  number1, number2)
            ABIAS_ann = abias_count(b_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]],  number1, number2)
            POD_ann = pod(b_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]],  number1, number2)
            FAR_ann = far(b_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]],  number1, number2)
            CSI_ann = csi(b_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]],  number1, number2)


            result_ann_cc[[i-1], [0]] = CC_ann
            result_ann_bias[[i-1], [0]] = BIAS_ann
            result_ann_rmse[[i-1], [0]] = RMSE_ann
            result_ann_mae[[i-1], [0]] = MAE_ann
            result_ann_abias[[i-1], [0]] = ABIAS_ann
            result_ann_pod[[i-1], [0]] = POD_ann
            result_ann_far[[i-1], [0]] = FAR_ann
            result_ann_csi[[i-1], [0]] = CSI_ann

            CC_final = cc_count(d_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]],  number1, number2)
            BIAS_final = bias_count(d_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]],  number1, number2)
            RMSE_final = rmse_count(d_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]], number1, number2)
            MAE_final = mae_count(d_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]],  number1, number2)
            ABIAS_final = abias_count(d_in[:, step[i - 1]:step[i]], c_in[:, step[i - 1]:step[i]], number1, number2)
            POD_final = pod(d_in[:, step[i - 1]:step[i]], c_in[:, step[i - 1]:step[i]], number1, number2)
            FAR_final = far(d_in[:, step[i - 1]:step[i]], c_in[:, step[i - 1]:step[i]], number1, number2)
            CSI_final = csi(d_in[:, step[i - 1]:step[i]], c_in[:, step[i - 1]:step[i]], number1, number2)

            result_final_cc[[i-1], [0]] = CC_final
            result_final_bias[[i-1], [0]] = BIAS_final
            result_final_rmse[[i-1], [0]] = RMSE_final
            result_final_mae[[i-1], [0]] = MAE_final
            result_final_abias[[i - 1], [0]] = ABIAS_final
            result_final_pod[[i - 1], [0]] = POD_final
            result_final_far[[i - 1], [0]] = FAR_final
            result_final_csi[[i - 1], [0]] = CSI_final


            # CC_nn = cc_count(e_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]],  number1, number2)
            # BIAS_nn = bias_count(e_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]],  number1, number2)
            # RMSE_nn = rmse_count(e_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]], number1, number2)
            # MAE_nn = mae_count(e_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]],  number1, number2)
            # ABIAS_nn = abias_count(e_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]],  number1, number2)
            # POD_nn = pod(e_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]],  number1, number2)
            # FAR_nn = far(e_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]],  number1, number2)
            # CSI_nn = csi(e_in[:,step[i-1]:step[i]], c_in[:,step[i-1]:step[i]],  number1, number2)
            #
            #
            # result_nn_cc[[i-1], [0]] = CC_nn
            # result_nn_bias[[i-1], [0]] = BIAS_nn
            # result_nn_rmse[[i-1], [0]] = RMSE_nn
            # result_nn_mae[[i-1], [0]] = MAE_nn
            # result_nn_abias[[i-1], [0]] = ABIAS_nn
            # result_nn_pod[[i-1], [0]] = POD_nn
            # result_nn_far[[i-1], [0]] = FAR_nn
            # result_nn_csi[[i-1], [0]] = CSI_nn


        path_save = 'F:\\毕业论文图\\青藏高原\\空间\\月份性能折线图\\'+ season +'\\'
        mkdir(path_save)  # 创建一个目录存放结果

        a = path_save + 'result_early_cc.txt'
        with open(a, 'w')as f:
            for i in range(len(result_early_cc)):
                for j in range(1):
                    f.writelines(str(result_early_cc[i, j]))
                    f.writelines('\n')
        a1 = path_save + 'result_early_bias.txt'
        with open(a1, 'w')as f:
            for i in range(len(result_early_bias)):
                for j in range(1):
                    f.writelines(str(result_early_bias[i, j]))
                    f.writelines('\n')
        a2 = path_save + 'result_early_rmse.txt'
        with open(a2, 'w')as f:
            for i in range(len(result_early_rmse)):
                for j in range(1):
                    f.writelines(str(result_early_rmse[i, j]))
                    f.writelines('\n')
        a3 = path_save + 'result_early_mae.txt'
        with open(a3, 'w')as f:
            for i in range(len(result_early_mae)):
                for j in range(1):
                    f.writelines(str(result_early_mae[i, j]))
                    f.writelines('\n')

        a4 = path_save + 'result_early_abias.txt'
        with open(a4, 'w')as f:
            for i in range(len(result_early_abias)):
                for j in range(1):
                    f.writelines(str(result_early_abias[i, j]))
                    f.writelines('\n')
        a5 = path_save + 'result_early_pod.txt'
        with open(a5, 'w')as f:
            for i in range(len(result_early_pod)):
                for j in range(1):
                    f.writelines(str(result_early_pod[i, j]))
                    f.writelines('\n')
        a6 = path_save + 'result_early_far.txt'
        with open(a6, 'w')as f:
            for i in range(len(result_early_far)):
                for j in range(1):
                    f.writelines(str(result_early_far[i, j]))
                    f.writelines('\n')
        a7 = path_save + 'result_early_csi.txt'
        with open(a7, 'w')as f:
            for i in range(len(result_early_csi)):
                for j in range(1):
                    f.writelines(str(result_early_csi[i, j]))
                    f.writelines('\n')

        b = path_save + 'result_final_cc.txt'
        with open(b, 'w')as f:
            for i in range(len(result_final_cc)):
                for j in range(1):
                    f.writelines(str(result_final_cc[i, j]))
                    f.writelines('\n')
        b1 = path_save + 'result_final_bias.txt'
        with open(b1, 'w')as f:
            for i in range(len(result_final_bias)):
                for j in range(1):
                    f.writelines(str(result_final_bias[i, j]))
                    f.writelines('\n')
        b2 = path_save + 'result_final_rmse.txt'
        with open(b2, 'w')as f:
            for i in range(len(result_final_rmse)):
                for j in range(1):
                    f.writelines(str(result_final_rmse[i, j]))
                    f.writelines('\n')
        b3 = path_save + 'result_final_mae.txt'
        with open(b3, 'w')as f:
            for i in range(len(result_final_mae)):
                for j in range(1):
                    f.writelines(str(result_final_mae[i, j]))
                    f.writelines('\n')

        b4 = path_save + 'result_final_abias.txt'
        with open(b4, 'w')as f:
            for i in range(len(result_final_abias)):
                for j in range(1):
                    f.writelines(str(result_final_abias[i, j]))
                    f.writelines('\n')
        b5 = path_save + 'result_final_pod.txt'
        with open(b5, 'w')as f:
            for i in range(len(result_final_pod)):
                for j in range(1):
                    f.writelines(str(result_final_pod[i, j]))
                    f.writelines('\n')
        b6 = path_save + 'result_final_far.txt'
        with open(b6, 'w')as f:
            for i in range(len(result_final_far)):
                for j in range(1):
                    f.writelines(str(result_final_far[i, j]))
                    f.writelines('\n')
        b7 = path_save + 'result_final_csi.txt'
        with open(b7, 'w')as f:
            for i in range(len(result_final_csi)):
                for j in range(1):
                    f.writelines(str(result_final_csi[i, j]))
                    f.writelines('\n')


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

        c4 = path_save + 'result_ann_abias.txt'
        with open(c4, 'w')as f:
            for i in range(len(result_ann_abias)):
                for j in range(1):
                    f.writelines(str(result_ann_abias[i, j]))
                    f.writelines('\n')
        c5 = path_save + 'result_ann_pod.txt'
        with open(c5, 'w')as f:
            for i in range(len(result_ann_pod)):
                for j in range(1):
                    f.writelines(str(result_ann_pod[i, j]))
                    f.writelines('\n')
        c6 = path_save + 'result_ann_far.txt'
        with open(c6, 'w')as f:
            for i in range(len(result_ann_far)):
                for j in range(1):
                    f.writelines(str(result_ann_far[i, j]))
                    f.writelines('\n')
        c7 = path_save + 'result_ann_csi.txt'
        with open(c7, 'w')as f:
            for i in range(len(result_ann_csi)):
                for j in range(1):
                    f.writelines(str(result_ann_csi[i, j]))
                    f.writelines('\n')

            # d = path_save + 'result_nn_cc.txt'
            # with open(d, 'w')as f:
            #     for i in range(len(result_nn_cc)):
            #         for j in range(1):
            #             f.writelines(str(result_nn_cc[i, j]))
            #             f.writelines('\n')
            # d1 = path_save + 'result_nn_bias.txt'
            # with open(d1, 'w')as f:
            #     for i in range(len(result_nn_bias)):
            #         for j in range(1):
            #             f.writelines(str(result_nn_bias[i, j]))
            #             f.writelines('\n')
            # d2 = path_save + 'result_nn_rmse.txt'
            # with open(d2, 'w')as f:
            #     for i in range(len(result_nn_rmse)):
            #         for j in range(1):
            #             f.writelines(str(result_nn_rmse[i, j]))
            #             f.writelines('\n')
            # d3 = path_save + 'result_nn_mae.txt'
            # with open(d3, 'w')as f:
            #     for i in range(len(result_nn_mae)):
            #         for j in range(1):
            #             f.writelines(str(result_nn_mae[i, j]))
            #             f.writelines('\n')
            #
            # d4 = path_save + 'result_nn_abias.txt'
            # with open(d4, 'w')as f:
            #     for i in range(len(result_nn_abias)):
            #         for j in range(1):
            #             f.writelines(str(result_nn_abias[i, j]))
            #             f.writelines('\n')
            # d5 = path_save + 'result_nn_pod.txt'
            # with open(d5, 'w')as f:
            #     for i in range(len(result_nn_pod)):
            #         for j in range(1):
            #             f.writelines(str(result_nn_pod[i, j]))
            #             f.writelines('\n')
            # d6 = path_save + 'result_nn_far.txt'
            # with open(d6, 'w')as f:
            #     for i in range(len(result_nn_far)):
            #         for j in range(1):
            #             f.writelines(str(result_nn_far[i, j]))
            #             f.writelines('\n')
            # d7 = path_save + 'result_nn_csi.txt'
            # with open(d7, 'w')as f:
            #     for i in range(len(result_nn_csi)):
            #         for j in range(1):
            #             f.writelines(str(result_nn_csi[i, j]))
            #             f.writelines('\n')

        # CC_early = cc_count(a_in[:,:], c_in[:,:], number1,number2)
        # print("CC_early :", CC_early)
        # BIAS_early = bias_count(a_in[:,:], c_in[:,:], number1,number2)
        # print("BIAS_early :", BIAS_early)
        # ABIAS_early = abias_count(a_in[:, :], c_in[:, :], number1, number2)
        # print("ABIAS_early :", ABIAS_early)
        # RMSE_early = rmse_count(a_in[:,:], c_in[:,:], number1,number2)
        # print("RMSE_early :", RMSE_early)
        # MAE_early = mae_count(a_in[:,:], c_in[:,:], number1,number2)
        # print("MAE_early :", MAE_early)
        # SD_early = SD_count(a_in[:,:], c_in[:,:], number1,number2)
        # print("SD_early :", SD_early)
        # POD_early = pod(a_in[:, :], c_in[:, :], number1, number2)
        # print("POD_early :", POD_early)
        # FAR_early = far(a_in[:, :], c_in[:, :], number1, number2)
        # print("FAR_early :",  FAR_early)
        # CSI_early = csi(a_in[:, :], c_in[:, :], number1, number2)
        # print("CSI_early :",  CSI_early)
        # print('*' * 10)
        #
        # CC_pre = cc_count(b_in[:,:], c_in[:,:], number1, number2)
        # print("CC_pre:", CC_pre)
        # BIAS_pre = bias_count(b_in[:,:], c_in[:,:], number1, number2)
        # print("BIAS_pre:", BIAS_pre)
        # ABIAS_pre = abias_count(b_in[:, :], c_in[:, :], number1, number2)
        # print("ABIAS_pre:", ABIAS_pre)
        # RMSE_pre = rmse_count(b_in[:,:], c_in[:,:], number1, number2)
        # print("RMSE_pre:", RMSE_pre)
        # MAE_pre = mae_count(b_in[:,:], c_in[:,:], number1, number2)
        # print("MAE_pre:", MAE_pre)
        # SD_pre = SD_count(b_in[:,:], c_in[:,:], number1, number2)
        # print("SD_pre :", SD_pre)
        # POD_pre = pod(b_in[:, :], c_in[:, :], number1, number2)
        # print("POD_pre :", POD_pre)
        # FAR_pre = far(b_in[:, :], c_in[:, :], number1, number2)
        # print("FAR_pre :", FAR_pre)
        # CSI_pre = csi(b_in[:, :], c_in[:, :], number1, number2)
        # print("CSI_pre :", CSI_pre)
        # print('*' * 10)
        #
        # CC_final = cc_count(d_in[:,:], c_in[:,:], number1,number2)
        # print("CC_final:", CC_final)
        # BIAS_final = bias_count(d_in[:,:], c_in[:,:], number1,number2)
        # print("BIAS_final:", BIAS_final)
        # ABIAS_final = abias_count(d_in[:, :], c_in[:, :], number1, number2)
        # print("ABIAS_final:", ABIAS_final)
        # RMSE_final = rmse_count(d_in[:,:], c_in[:,:], number1,number2)
        # print("RMSE_final:", RMSE_final)
        # MAE_final = mae_count(d_in[:,:], c_in[:,:], number1,number2)
        # print("MAE_final:", MAE_final)
        # SD_final = SD_count(d_in[:,:], c_in[:,:], number1,number2)
        # print("SD_final:", SD_final)
        # POD_final = pod(d_in[:,:], c_in[:,:], number1,number2)
        # print("POD_final:",  POD_final)
        # FAR_final = far(d_in[:,:], c_in[:,:], number1,number2)
        # print("FAR_final:",  FAR_final)
        # CSI_final = csi(d_in[:,:], c_in[:,:], number1,number2)
        # print("CSI_final:",  CSI_final)
        # print('*' * 10)


        # CC_NN = cc_count(e_in[:,:], c_in[:,:], number1,number2)
        # print("CC_NN:", CC_NN)
        # BIAS_NN = bias_count(e_in[:,:], c_in[:,:], number1,number2)
        # print("BIAS_NN:", BIAS_NN)
        # ABIAS_NN = abias_count(e_in[:, :], c_in[:, :], number1, number2)
        # print("ABIAS_NN:", ABIAS_NN)
        # RMSE_NN = rmse_count(e_in[:,:], c_in[:,:], number1,number2)
        # print("RMSE_NN:", RMSE_NN)
        # MAE_NN = mae_count(e_in[:,:], c_in[:,:], number1,number2)
        # print("MAE_NN:", MAE_NN)
        # SD_NN = SD_count(e_in[:,:], c_in[:,:], number1,number2)
        # print("SD_NN:", SD_NN)
        # POD_NN = pod(e_in[:,:], c_in[:,:], number1,number2)
        # print("POD_NN:",  POD_NN)
        # FAR_NN = far(e_in[:,:], c_in[:,:], number1,number2)
        # print("FAR_NN:",  FAR_NN)
        # CSI_NN = csi(e_in[:,:], c_in[:,:], number1,number2)
        # print("CSI_NN:",  CSI_NN)
        # print('*' * 10)


