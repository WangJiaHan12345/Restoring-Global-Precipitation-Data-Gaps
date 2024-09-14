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
    return filelist  # for filename in filelist



# path_dem_csv = "H:\\时间预测\\四个区域可用数据\\shirun\\dem\\DEM.csv"
# data_d = read_csv(path_dem_csv)
# data_dem = data_d[:, 1:]
# # print(data_dem.shape)  # （44，41）
# rows = data_dem.shape[0]
# cols = data_dem.shape[1]
#
# file_txt = open('H:\\时间预测\\四个区域可用数据\\shirun\\class_sumrain.txt')
# lines = file_txt.readlines()
# data_class = []
# for line in lines:
#     data_class.append(line.strip('\n'))
# print('data:', data_class)
#
# class_data = np.zeros((240,720)).astype('int64')
# i = 0
# for row in range(rows):
#     for col in range(cols):
#         if data_dem[row, col] == -9999:
#             class_data[row, col] = -9999
#         else:
#             class_data[row, col] = data_class[i]
#             i += 1
# print(class_data)
#
# data1 = pd.DataFrame(class_data)
# data1.to_csv('H:\\时间预测\\四个区域可用数据\\shirun\\class_sumrain.csv')



path_p = 'H:\\青藏高原数据\\时间预测\\2015-2016(final)\\02_season_data\\xunlian\\gsmap_gauge\\3-5\\'  #station_txt就是cpc
    # path_p = 'H:\\ganhan\\01_clip_data\\cpc'
p_name = read_dir(path_p)


file_txt = open('H:\\青藏高原数据\\时间预测\\2015-2016(final)\\class_sumrain_12-2.txt')
lines = file_txt.readlines()
data_class = []
for line in lines:
    data_class.append(line.strip('\n'))

class_data = np.zeros((440,700)).astype('int64')
# i = 0
# for row in range(rows):
#     for col in range(cols):
#         if data_dem[row, col] == -9999:
#             class_data[row, col] = -9999
#         else:
#             class_data[row, col] = data_class[i]
#             i += 1
# print(class_data)

for row in range(440):
    for col in range(700):
        class_data[row, col] = -9999

count = 0
for p in p_name:
    i = int(p[0:3])
    j = int(p[3:6])
    class_data[i - 1][j - 1] = data_class[count]
    count += 1
print('i',count)

data1 = pd.DataFrame(class_data)
data1.to_csv('H:\\青藏高原数据\\时间预测\\2015-2016(final)\\class_sumrain_12-2.csv')





