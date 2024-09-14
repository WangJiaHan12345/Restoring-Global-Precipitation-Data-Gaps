# -*- coding: utf-8 -*-
"""
将txt里面逐个数据提取出来按map存入excel文件
"""
import pandas as pd
import numpy as np
import os


def read_csv(path_csv):
    df = pd.read_csv(path_csv)
    data = np.array(df)
    return data


def read_dir(path):
    # 获取文件夹下所有txt文件名
    filelist = os.listdir(path)
    return filelist


path_dem_csv = "H:\\时间预测\\四个区域可用数据\\shirun\\dem\\DEM.csv"
data_d = read_csv(path_dem_csv)
#北半球
# data_dem = data_d[:120, 1:]
#南半球
data_dem = data_d[120:, 1:]


rows = data_dem.shape[0]
cols = data_dem.shape[1]
print(rows,cols)


file_txt = open('H:\\时间预测\\四个区域可用数据\\shirun\\class_sumrain_12-2_n_shiyan.txt')
lines = file_txt.readlines()
data_class = []
for line in lines:
    data_class.append(line.strip('\n'))

class_data = np.zeros((229,720)).astype('int64')  #北是240 南是具体的行数


#用来决定哪些网格是可用的
path_p_used = 'H:\\时间预测\\四个区域可用数据\\shirun\\02_grid_data\\xunlian4\\cpc4\\南\\12-2'  # station_txt就是cpc
p_name_used = read_dir(path_p_used)



i = 0

#北半球
# sum=240
# for row in range(rows):
#     for col in range(cols):
#         temp = str(row+1).zfill(3) + str(col+1).zfill(3) + '.txt'
#         if temp in p_name_used:
#             class_data[row, col] = data_class[i]
#             i += 1
#         else:
#             class_data[row, col] = -9999
#
# for row in range(rows,sum):
#     for col in range(cols):
#             class_data[row, col] = -9999

#南半球
num=120
for row in range(num):
    for col in range(cols):
            class_data[row, col] = -9999
for row in range(num,rows+num):
    for col in range(cols):
        temp = str(row + 1).zfill(3) + str(col + 1).zfill(3) + '.txt'
        if temp in p_name_used:
            class_data[row, col] = data_class[i]
            i += 1
        else:
            class_data[row, col] = -9999


print(class_data)

data1 = pd.DataFrame(class_data)
data1.to_csv('H:\\时间预测\\四个区域可用数据\\shirun\\class_sumrain_12-2_n_shiyan.cvs')







