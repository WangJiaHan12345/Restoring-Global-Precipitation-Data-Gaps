# -*- coding: utf-8 -*-
"""
data_time2grid  经过一系列筛选得到最终有使用意义的数据

@author: Zzy
"""
import pandas as pd
import numpy as np
import math
import os
import csv


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


def read_datas(file_dir):
    data = []
    for root, dirs, files in os.walk(file_dir):
        data.append(files)
    return data


def read_csv(path_csv):
    df = pd.read_csv(path_csv)
    data = np.array(df)
    return data


def grid_name(data_dem,rows, cols):
    # 获取有dem并且网格站点数>=1的网格编号
    grid_row = []
    grid_col = []
    for row in range(rows):
        for col in range(cols):
            if data_dem[row][col] != -9999:
            # if data_dem[row][col] != -9999:
                grid_row.append("{:03d}".format(row + 1))  # 要+1，变成行列的数字，而不是从0开始
                grid_col.append("{:03d}".format(col + 1))
    return grid_row, grid_col


def read_dir(path):
    # 获取文件夹下所有txt文件名
    filelist = os.listdir(path)
    return filelist


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


def txt_create(path_txt, num1, num2, val, name):
    # 创建txt,并追加写入数据
    file = open(path_txt + '\\' + name + "_{}{}.txt".format(num1, num2), 'a+')  # num1:list_row[],num2:list_col[]
    file.write("{:.4f}".format(val) + '\n')


if __name__ == '__main__':   #主程序开始

    data_path = 'H:\\GSMAP数据\\读取后的数据-0.5\\gsmap_mvk\\'
    #与3-5无关  此路径只是起到一个告诉可用网格的作用
    standard1_path = 'H:\\空间预测\\shirun\\02_final_data\\cpc4\\3-5\\'  # 名称3-5_001011
    standard2_path = 'H:\\GSMAP数据\\15-18网格数据-全年\\gsmap_mvk\\shirun\\'  # 输出位置


    data_names = read_datas(data_path)[0]
    standard1_names = read_datas(standard1_path)[0]
    standard2_names = read_datas(standard2_path)[0]


