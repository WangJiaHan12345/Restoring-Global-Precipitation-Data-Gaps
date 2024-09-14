# -*- coding: utf-8 -*-
"""
96 hours average precipitation
rain txt to csv

@author: Zzy
"""
import pandas as pd
import numpy as np
import math
import os
import csv


def read_dir(path):
    # 获取文件夹下所有txt文件名
    filelist = os.listdir(path)
    return filelist  # for filename in filelist


def read_txt(file_name):
    # 读取txt数据，以矩阵形式
    file_txt = open(file_name)
    lines = file_txt.readlines()

    data = []
    for line in lines:
        data.append(line.strip('\n'))

    data_ = []
    for i in range(6, len(data)):
        data_.append(data[i].strip(' ').split(' '))
    data_ = np.array(data_)
    data_ = data_.astype('float64')
    return data_

def read_grid(file_name):
    # 读取txt数据，以矩阵形式
    data = 0
    with open(file_name,'r') as f:
        for line in f:
            data = data + float(line)
    return data



if __name__ == '__main__':

    path_p = 'H:\\时间预测\\四个区域可用数据\\ganhan\\只选用有地面站点\\02_grid_data\\xunlian4\\cpc\\12-2\\'  #station_txt就是cpc
    p_name = read_dir(path_p)

    data_all = np.zeros((240,720)).astype('float64')
    data_all.fill(-9999)

#原始数据是天数的格式
    # for p in p_name:
    #     file_name = path_p + '\\' + p
    #     data = read_txt(file_name)
    #     data_all += data      #各站点96小时的降水量各自的总量


#原始数据是网格的格式
    for p in p_name:
        file_name = path_p + '\\' + p
        data = read_grid(file_name)
        i = int(p[0:3])
        j = int(p[3:6])
        data_all[i-1][j-1] = data


    data1 = pd.DataFrame(data_all)
    # data1[data1 < 0] = -9999
    data1.to_csv('G:\\全球\\时间预测结果\\画相对图所用数据\\季度\\原始数据_1\\分气候区\\干旱区\\经纬度相关性\\rain_sum_12-2.csv')
