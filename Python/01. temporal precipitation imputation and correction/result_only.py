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
from itertools import chain
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


def SD_count(x_in, number1, number2):
    # 求标准差
    sum = 0
    count = 0
    average_s = np.mean(x_in)
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

def me_count(x_in, s_in, number1, number2):
    sum = 0
    count = 0
    for i in range(number1):
        for j in range(number2):
            count += 1
            cha = x_in[i][j] - s_in[i][j]
            sum += cha
    me = sum / count
    return me

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


def pod(x_in, s_in, number1, number2):  #如果是一个月则用72
    # 求pod
    n10 = 0
    n01 = 0
    n11 = 0
    for j in range(number1):
        for k in range(number2):
        # CPC
            if s_in[j][k] >=2.4 and x_in[j][k] <2.4:
                n01 = n01 + 1
            # 卫星
            if x_in[j][k] >=2.4 and s_in[j][k] < 2.4:
                n10 = n10 + 1
            #     cpc+卫星
            if x_in[j][k] >=2.4 and s_in[j][k] >=2.4:
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
            if s_in[j][k] >=2.4 and x_in[j][k] <2.4:
                n01 = n01 + 1
            # 卫星
            if x_in[j][k] >=2.4 and s_in[j][k] <2.4:
                n10 = n10 + 1
            #     cpc+卫星
            if x_in[j][k] >=2.4 and s_in[j][k] >=2.4:
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
            if s_in[j][k] >=2.4 and x_in[j][k] <2.4:
                n01 = n01 + 1
            # 卫星
            if x_in[j][k] >=2.4 and s_in[j][k] <2.4:
                n10 = n10 + 1
            #     cpc+卫星
            if x_in[j][k] >=2.4 and s_in[j][k] >=2.4:
                n11 = n11 + 1

    csi = n11/(n11+n10+n01)
    return csi

banqius = ['标准季节']
for banqiu in banqius:
    nums = ['9-11']
    for num in nums:
        if __name__ == '__main__':
            # a_path = 'G:\\全球\\时间预测结果\\'+ banqiu +'\\只有站点网格\\1-12\\Early\\'
            # b_path = 'G:\\全球\\时间预测结果\\'+ banqiu +'\\只有站点网格\\1-12\\ANN\\'
            # c_path = 'G:\\全球\\时间预测结果\\'+ banqiu +'\\只有站点网格\\1-12\\cpc\\'
            # d_path = 'G:\\全球\\时间预测结果\\'+ banqiu +'\\只有站点网格\\1-12\\Final\\'

            # a_path = 'G:\\全球\\空间校正结果\\global_ns\\'+ banqiu +'\\标准季节\\'+ num +'\\Early\\'
            # b_path = 'G:\\全球\\空间校正结果\\global_ns\\'+ banqiu +'\\标准季节\\'+ num +'\\ANN\\'
            # c_path = 'G:\\全球\\空间校正结果\\global_ns\\'+ banqiu +'\\标准季节\\'+ num +'\\cpc\\'
            # d_path = 'G:\\全球\\空间校正结果\\global_ns\\'+ banqiu +'\\标准季节\\'+ num +'\\Final\\'
            # e_path = 'G:\\全球\\空间校正结果\\global_ns\\'+ banqiu +'\\标准季节\\'+ num +'\\DNN\\'

            # a_path = 'G:\\青藏高原\\时间\\季度\\'+ banqiu +'\\'+ num +'\\mvk\\'
            # b_path = 'G:\\青藏高原\\时间\\季度\\'+ banqiu +'\\'+ num +'\\ANN\\'
            # c_path = 'G:\\青藏高原\\时间\\季度\\'+ banqiu +'\\'+ num +'\\国家气象局\\'
            # d_path = 'G:\\青藏高原\\时间\\季度\\'+ banqiu +'\\'+ num +'\\gauge\\'

            # a_path = 'G:\\青藏高原\\空间\\'+ banqiu +'\\'+ num +'\\gsmap_mvk\\'
            # b_path = 'G:\\青藏高原\\空间\\'+ banqiu +'\\'+ num +'\\ANN\\'
            # c_path = 'G:\\青藏高原\\空间\\'+ banqiu +'\\'+ num +'\\国家气象局\\'
            # d_path = 'G:\\青藏高原\\空间\\'+ banqiu +'\\'+ num +'\\gsmap_gauge\\'

            # a_path = 'G:\\全球\\空间校正结果\\用于和GPCP比较\\月尺度\\最终可用数据\\Early_2.5\\'
            # b_path = 'G:\\全球\\空间校正结果\\用于和GPCP比较\\月尺度\\最终可用数据\\ANN_2.5\\'
            # c_path = 'G:\\全球\\空间校正结果\\用于和GPCP比较\\月尺度\\最终可用数据\\GPCP_2.5\\'
            # d_path = 'G:\\全球\\空间校正结果\\用于和GPCP比较\\月尺度\\最终可用数据\\Final_2.5\\'

            # a_path = 'G:\\全球\\时间预测结果\\地面站点_Early\\global_ns\\标准季节\\'+num+'\\early\\'
            # b_path = 'G:\\全球\\时间预测结果\\地面站点_Early\\global_ns\\标准季节\\'+num+'\\ANN\\'
            # c_path = 'G:\\全球\\时间预测结果\\地面站点_Early\\global_ns\\标准季节\\'+num+'\\cpc\\'
            # d_path = 'G:\\全球\\时间预测结果\\地面站点_Early\\global_ns\\标准季节\\'+num+'\\final\\'


            a_path = 'G:\\全球\\空间校正结果\\用于和GPCP比较\\月尺度\\更小范围内的\\Final_2.5\\'
            c_path = 'G:\\全球\\空间校正结果\\用于和GPCP比较\\月尺度\\更小范围内的\\GPCP_2.5\\'
            # c_path = 'G:\\青藏高原\\时间\\季度\\只包含站点的网格(rnt)\\综合\\国家气象局\\'
            # d_path = 'G:\\青藏高原\\时间\\季度\\只包含站点的网格(rnt)\\综合\\gauge\\'
            # e_path = 'G:\\全球\\空间校正结果\\global_ns\\有DNN\\标准季节\\'+num+'\\DNN\\'

            # a_path = 'G:\\全球\\时间预测结果\\地面站点_final\\'+num+'\\总\\1-12\\Early\\'
            # b_path = 'G:\\全球\\时间预测结果\\地面站点_final\\'+num+'\\总\\1-12\\ANN\\'
            # c_path = 'G:\\全球\\时间预测结果\\地面站点_final\\'+num+'\\总\\1-12\\cpc\\'
            # d_path = 'G:\\全球\\时间预测结果\\地面站点_final\\'+num+'\\总\\1-12\\Final\\'
            # e_path = 'G:\\全球\\时间预测结果\\地面站点_final\\'+num+'\\总\\1-12\\ANN_差\\'
            # f_path = 'G:\\全球\\时间预测结果\\地面站点_final\\'+num+'\\总\\1-12\\ANN_最差\\'

            a_names = read_datas(a_path)[0]
            # b_names = read_datas(b_path)[0]
            c_names = read_datas(c_path)[0]
            # d_names = read_datas(d_path)[0]
            # e_names = read_datas(e_path)[0]
            # f_names = read_datas(f_path)[0]


            a_in = [] #用来存放各个mvk里的数据
            # b_in = []
            c_in = []
            # d_in = []
            # e_in = []
            # f_in = []


            a_in = save_datas(a_path, a_names, a_in)
            # b_in = save_datas(b_path,b_names,b_in)
            c_in = save_datas(c_path, c_names, c_in)
            # d_in = save_datas(d_path,d_names,d_in)
            # e_in = save_datas(e_path, e_names, e_in)
            # f_in = save_datas(f_path, f_names, f_in)

            # number1 = c_in.shape[0]
            # number2 = c_in.shape[1]
            # print(number1,number2)

            a_in = np.array(a_in)
            # a_in = a_in.astype('float64')
            a_in = list(chain.from_iterable(a_in))
            a_in = np.array([a_in]).T
            a_in = a_in.astype('float64')

            #
            # b_in = np.array(b_in)
            # # b_in = b_in.astype('float64')
            # b_in = list(chain.from_iterable(b_in))
            # b_in = np.array([b_in]).T
            # b_in = b_in.astype('float64')

            c_in = np.array(c_in)
            # c_in = c_in.astype('float64')
            c_in = list(chain.from_iterable(c_in))
            c_in = np.array([c_in]).T
            c_in = c_in.astype('float64')


            #
            # d_in = np.array(d_in)
            # # d_in = d_in.astype('float64')
            # d_in = list(chain.from_iterable(d_in))
            # d_in = np.array([d_in]).T
            # d_in = d_in.astype('float64')

            # e_in = np.array(e_in)
            # # e_in = e_in.astype('float64')
            # e_in = list(chain.from_iterable(e_in))
            # e_in = np.array([e_in]).T
            # e_in = e_in.astype('float64')

            # f_in = np.array(f_in)
            # # e_in = e_in.astype('float64')
            # f_in = list(chain.from_iterable(f_in))
            # f_in = np.array([f_in]).T
            # f_in = f_in.astype('float64')


            number1 = len(c_in)
            number2 = 1

            print(banqiu)
            print(num)
            CC_early = cc_count(a_in[:,:], c_in[:,:], number1,number2)
            print("CC_nn:", CC_early)
            BIAS_early = bias_count(a_in[:,:], c_in[:,:], number1,number2)
            print("BIAS_nn:", BIAS_early)
            ABIAS_early = abias_count(a_in[:, :], c_in[:, :], number1, number2)
            print("ABIAS_nn :", ABIAS_early)
            RMSE_early = rmse_count(a_in[:,:], c_in[:,:], number1,number2)
            print("RMSE_nn:", RMSE_early)
            MAE_early = mae_count(a_in[:,:], c_in[:,:], number1,number2)
            print("MAE_nn :", MAE_early)
            ME_early = me_count(a_in[:,:], c_in[:,:], number1,number2)
            print("ME_nn :", ME_early)
            SD_early = SD_count(a_in[:,:], number1,number2)
            print("SD_nn :", SD_early)
            POD_early = pod(a_in[:, :], c_in[:, :], number1, number2)
            print("POD_nn:", POD_early)
            FAR_early = far(a_in[:, :], c_in[:, :], number1, number2)
            print("FAR_nn :",  FAR_early)
            CSI_early = csi(a_in[:, :], c_in[:, :], number1, number2)
            print("CSI_nn :",  CSI_early)
            print('*' * 10)

            # CC_pre = cc_count(b_in[:,:], c_in[:,:], number1, number2)
            # print("CC_ann_nn:", CC_pre)
            # BIAS_pre = bias_count(b_in[:,:], c_in[:,:], number1, number2)
            # print("BIAS_ann_nn:", BIAS_pre)
            # ABIAS_pre = abias_count(b_in[:, :], c_in[:, :], number1, number2)
            # print("ABIAS_ann_nn:", ABIAS_pre)
            # RMSE_pre = rmse_count(b_in[:,:], c_in[:,:], number1, number2)
            # print("RMSE_ann_nn:", RMSE_pre)
            # MAE_pre = mae_count(b_in[:,:], c_in[:,:], number1, number2)
            # print("MAE_ann_nn:", MAE_pre)
            # ME_pre = me_count(b_in[:, :], c_in[:, :], number1, number2)
            # print("ME_ann_nn:", ME_pre)
            # SD_pre = SD_count(b_in[:,:], number1, number2)
            # print("SD_ann_nn :", SD_pre)
            # SD_obs = SD_count(c_in[:,:], number1, number2)
            # print("SD_obs :", SD_obs)
            # POD_pre = pod(b_in[:, :], c_in[:, :], number1, number2)
            # print("POD_ann_nn:", POD_pre)
            # FAR_pre = far(b_in[:, :], c_in[:, :], number1, number2)
            # print("FAR_ann_nn :", FAR_pre)
            # CSI_pre = csi(b_in[:, :], c_in[:, :], number1, number2)
            # print("CSI_ann_nn :", CSI_pre)
            # print('*' * 10)

            # CC_final = cc_count(d_in[:,:], c_in[:,:], number1,number2)
            # print("CC_gauge:", CC_final)
            # BIAS_final = bias_count(d_in[:,:], c_in[:,:], number1,number2)
            # print("BIAS_gauge:", BIAS_final)
            # ABIAS_final = abias_count(d_in[:, :], c_in[:, :], number1, number2)
            # print("ABIAS_gauge:", ABIAS_final)
            # RMSE_final = rmse_count(d_in[:,:], c_in[:,:], number1,number2)
            # print("RMSE_gauge:", RMSE_final)
            # MAE_final = mae_count(d_in[:,:], c_in[:,:], number1,number2)
            # print("MAE_gauge:", MAE_final)
            # ME_final = me_count(d_in[:, :], c_in[:, :], number1, number2)
            # print("ME_gauge:", ME_final)
            # SD_final = SD_count(d_in[:,:], number1,number2)
            # print("SD_gauge:", SD_final)
            # POD_final = pod(d_in[:,:], c_in[:,:], number1,number2)
            # print("POD_gauge:",  POD_final)
            # FAR_final = far(d_in[:,:], c_in[:,:], number1,number2)
            # print("FAR_gauge:",  FAR_final)
            # CSI_final = csi(d_in[:,:], c_in[:,:], number1,number2)
            # print("CSI_gauge:",  CSI_final)
            # print('*' * 10)


            # CC_NN = cc_count(e_in[:,:], c_in[:,:], number1,number2)
            # print("CC_gauge_NN:", CC_NN)
            # BIAS_NN = bias_count(e_in[:,:], c_in[:,:], number1,number2)
            # print("BIAS_gauge_NN:", BIAS_NN)
            # ABIAS_NN = abias_count(e_in[:, :], c_in[:, :], number1, number2)
            # print("ABIAS_gauge_NN:", ABIAS_NN)
            # RMSE_NN = rmse_count(e_in[:,:], c_in[:,:], number1,number2)
            # print("RMSE_gauge_NN:", RMSE_NN)
            # MAE_NN = mae_count(e_in[:,:], c_in[:,:], number1,number2)
            # print("MAE_gauge_NN:", MAE_NN)
            # ME_NN = me_count(e_in[:, :], c_in[:, :], number1, number2)
            # print("ME_gauge_NN:", ME_NN)
            # SD_NN = SD_count(e_in[:,:], number1,number2)
            # print("SD_gauge_NN:", SD_NN)
            # POD_NN = pod(e_in[:,:], c_in[:,:], number1,number2)
            # print("POD_gauge_NN:",  POD_NN)
            # FAR_NN = far(e_in[:,:], c_in[:,:], number1,number2)
            # print("FAR_gauge_NN:",  FAR_NN)
            # CSI_NN = csi(e_in[:,:], c_in[:,:], number1,number2)
            # print("CSI_gauge_NN:",  CSI_NN)
            # print('*' * 10)


            # CC_NN = cc_count(f_in[:,:], c_in[:,:], number1,number2)
            # print("CC_mvk:", CC_NN)
            # BIAS_NN = bias_count(f_in[:,:], c_in[:,:], number1,number2)
            # print("BIAS_mvk:", BIAS_NN)
            # ABIAS_NN = abias_count(f_in[:, :], c_in[:, :], number1, number2)
            # print("ABIAS_mvk:", ABIAS_NN)
            # RMSE_NN = rmse_count(f_in[:,:], c_in[:,:], number1,number2)
            # print("RMSE_mvk:", RMSE_NN)
            # MAE_NN = mae_count(f_in[:,:], c_in[:,:], number1,number2)
            # print("MAE_mvk:", MAE_NN)
            # ME_NN = me_count(f_in[:,:], c_in[:,:], number1, number2)
            # print("ME_mvk:", ME_NN)
            # SD_NN = SD_count(f_in[:,:],number1,number2)
            # print("SD_mvk:", SD_NN)
            # POD_NN = pod(f_in[:,:], c_in[:,:], number1,number2)
            # print("POD_mvk:",  POD_NN)
            # FAR_NN = far(f_in[:,:], c_in[:,:], number1,number2)
            # print("FAR_mvk:",  FAR_NN)
            # CSI_NN = csi(f_in[:,:], c_in[:,:], number1,number2)
            # print("CSI_mvk:",  CSI_NN)
            # print('*' * 10)
