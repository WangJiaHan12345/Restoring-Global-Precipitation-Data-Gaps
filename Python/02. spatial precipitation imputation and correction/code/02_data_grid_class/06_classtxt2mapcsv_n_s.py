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


path_dem_csv = "H:\\banganhan\\dem\\global\\DEM.csv"
data_d = read_csv(path_dem_csv)
data_dem = data_d[:, 1:]
#北半球
# data_dem_n = data_d[:120, 1:]
#南半球
data_dem_s = data_d[120:, 1:]

#北半球
# rows = data_dem_n.shape[0]
# cols = data_dem_n.shape[1]
# 南半球
rows = data_dem_s.shape[0]
cols = data_dem_s.shape[1]
print(rows,cols)

# file_txt = open('H:\\banganhan\\Early_n\\class_sumrain_12-2.txt')
file_txt = open('H:\\banganhan\\Early_s\\class_sumrain_12-2.txt')
lines = file_txt.readlines()
data_class = []
for line in lines:
    data_class.append(line.strip('\n'))
print('data:', data_class)

class_data = np.zeros((231,720)).astype('int64')

i = 0

#北半球
# sum=231
# for row in range(rows):
#     for col in range(cols):
#         if data_dem[row, col] == -9999:
#             class_data[row, col] = -9999
#         else:
#             class_data[row, col] = data_class[i]
#             i += 1
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
        if data_dem[row, col] == -9999:
            class_data[row, col] = -9999
        else:
            class_data[row, col] = data_class[i]
            i += 1

print(class_data)

data1 = pd.DataFrame(class_data)
# data1.to_csv('H:\\banganhan\\Early_n\\class_sumrain_12-2.csv')
data1.to_csv('H:\\banganhan\\Early_s\\class_sumrain_12-2.csv')







