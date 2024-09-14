# # -*- coding: utf-8 -*-
# """
# 将txt里面数据提取出来存入csv文件
# """
# import xlrd
# import xlwt
# import pandas as pd
# import numpy as np
# import os
#
#
# file_txt = open('H:\\青藏高原数据\\rastert_extract1.txt')
# lines = file_txt.readlines()
#
#
# data = []
# for line in lines:
#     data.append(line.strip('\n'))
# # print(data)
# data_ = []
# for i in range(6, len(data)):  #从6开始是指去掉了ncols nrows xllcorner yllcorner cellsize  NODATA_value
#     data_.append(data[i].strip(' ').split(' '))
# # print(data_)
#
#
# data_ = np.array(data_)
# # data_ = data_.astype('float64')  # 先将字符串数组转成浮点型，才可继续变整型
# # data_ = data_.astype('int64')   #注意 dem  rain 一个有一个没有
#
# data1 = pd.DataFrame(data_)
# data1.to_csv('H:\\青藏高原数据\\rastert_extract1.csv')




# excle转txt
import pandas as pd

df = pd.read_excel('G:\\全球\\时间预测结果\\画相对图所用数据\\温度\\T_sum.xlsx', sheet_name='T_sum', header=None)		# 使用pandas模块读取数据
print('开始写入txt文件...')
df.to_csv('G:\\全球\\时间预测结果\\画相对图所用数据\\温度\\T_sum.txt', header=None, sep=' ', index=False)		# 写入，逗号分隔
print('文件写入成功!')


