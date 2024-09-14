# -*- coding: utf-8 -*-
"""
区域分块复制出来ABC
"""
import pandas as pd
import numpy as np
import math
import os
import csv
import shutil


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


def read_csv(path_csv):
    df = pd.read_csv(path_csv)
    data = np.array(df)
    return data


def read_txt(path_txt):
    # 读取文件名
    filelist = os.listdir(path_txt)
    # print(filelist)
    #去除前缀
    # file_names = np.array([file.split('_')[1] for file in filelist if file.endswith('.txt')], dtype=object)
    #去除.txt
    names = np.array([name.split('.')[0] for name in filelist if name.endswith('.txt')], dtype=object)
    return filelist,names

path_class_csv = 'H:\\青藏高原数据\\时间预测\\2015-2016\\class_sumrain_12-2.csv'
data_c = read_csv(path_class_csv)
data_class = data_c[:, 1:]


# earth = '北'
kinds = ['训练', '测试']
for kind in kinds:
    season = '12-2'
    names = ['高程']
    for name in names:
        path_txt ='H:\\青藏高原数据\\时间预测\\2015-2016\\01_clip_data\\'+ kind +'\\'+ name +'\\'+ season +'\\' #改日期
        filelist,file_names = read_txt(path_txt)


        path_save_0 = 'H:\\青藏高原数据\\时间预测\\2015-2016\\02_final_data\\class_data_sumrain_'+ season +'\\' + kind +'\\'+ name +'\\A\\'
        mkdir(path_save_0)
        path_save_1 = 'H:\\青藏高原数据\\时间预测\\2015-2016\\02_final_data\\class_data_sumrain_'+ season +'\\' + kind +'\\'+ name +'\\B\\'
        mkdir(path_save_1)
        path_save_2 = 'H:\\青藏高原数据\\时间预测\\2015-2016\\02_final_data\\class_data_sumrain_'+ season +'\\' + kind +'\\'+ name +'\\C\\'
        mkdir(path_save_2)
        path_save_3 = 'H:\\青藏高原数据\\时间预测\\2015-2016\\02_final_data\\class_data_sumrain_'+ season +'\\' + kind +'\\'+ name +'\\D\\'
        mkdir(path_save_3)
        path_save_4 = 'H:\\青藏高原数据\\时间预测\\2015-2016\\02_final_data\\class_data_sumrain_'+ season +'\\' + kind +'\\'+ name +'\\E\\'
        mkdir(path_save_4)
        path_save_5 = 'H:\\青藏高原数据\\时间预测\\2015-2016\\02_final_data\\class_data_sumrain_'+ season +'\\' + kind +'\\'+ name +'\\F\\'
        mkdir(path_save_5)
        # path_save_6 = 'H:\\青藏高原数据\\时间预测\\2015-2016\\02_final_data\\class_data_sumrain_'+ season +'\\' + kind +'\\'+ name +'\\G\\'
        # mkdir(path_save_6)
        # path_save_7 = 'H:\\时间预测\\四个区域可用数据\\banganhan\\03_class_data\\季节\\' + earth + '\\' + kind +'\\'+ name + '\\'+ season + '\\H\\'
        # mkdir(path_save_7)

        for i in range(len(filelist)):
            row = int(file_names[i][0:3]) - 1
            col = int(file_names[i][3:]) - 1
            path_scr = path_txt + '\\' + filelist[i]
            if data_class[row, col] == 0:
                # 复制到0文件夹
                shutil.copy(path_scr, path_save_0)
            elif data_class[row, col] == 1:
                # 复制到1文件夹
                shutil.copy(path_scr, path_save_1)
            elif data_class[row, col] == 2:
                # 复制到2文件夹
                shutil.copy(path_scr, path_save_2)
            elif data_class[row, col] == 3:
                    # 复制到3文件夹
                    shutil.copy(path_scr, path_save_3)
            elif data_class[row, col] == 4:
                    # 复制到3文件夹
                    shutil.copy(path_scr, path_save_4)
            elif data_class[row, col] == 5:
                    # 复制到3文件夹
                    shutil.copy(path_scr, path_save_5)
            elif data_class[row, col] == 6:
                    # 复制到3文件夹
                    shutil.copy(path_scr, path_save_6)
            # elif data_class[row, col] == 7:
            #         # 复制到3文件夹
            #         shutil.copy(path_scr, path_save_7)
            else:
                     continue
