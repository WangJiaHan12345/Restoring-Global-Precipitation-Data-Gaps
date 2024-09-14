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


def txt_create(path_txt, num1, num2, val, name):
    # 创建txt,并追加写入数据
    file = open(path_txt + '\\' + name + "_{}{}.txt".format(num1, num2), 'a+')  # num1:list_row[],num2:list_col[]
    file.write("{:.4f}".format(val) + '\n')


if __name__ == '__main__':   #主程序开始
    # path_station_csv = "H:\\5\\station\\station.csv"
    path_dem_csv = "E:\\ganhan\\dem\\DEM.csv"



    data_d = read_csv(path_dem_csv)
    data_dem = data_d[:, 1:]
    rows = data_dem.shape[0]
    cols = data_dem.shape[1]

    grid_row, grid_col = grid_name(data_dem,rows, cols)
    print(grid_row,grid_col)

    name = '6-8'
    path_p = 'E:\\ganhan\\01_clip_data\\Early4\\' + name  # cpc4  early4  final4
    p_name = read_dir(path_p)
    # print(p_name)

    path_save = 'E:\\ganhan\\02_final_data\\Early4\\' + name
    mkdir(path_save)
    print(len(grid_row))

    for i in range(len(grid_row)):
        num1 = grid_row[i]  # str 02
        num2 = grid_col[i]  # str 31
        for p in p_name:
            file_name = path_p + '\\' + p
            data = read_txt(file_name)
            val = data[int(num1)-1, int(num2)-1]
            if float(val) == -9999:
                break
            else:
                txt_create(path_save, num1, num2, val, name)


####################################################################################################################

# import pandas as pd
# import numpy as np
# import math
# import os
# import csv
#
#
# def mkdir(path):
#     # 去除首位空格
#     path = path.strip()
#     # 去除尾部 \ 符号
#     path = path.rstrip("\\")
#     # 判断路径是否存在
#     # 存在     True
#     # 不存在   False
#     isExists = os.path.exists(path)
#     # 判断结果
#     if not isExists:
#         # 如果不存在则创建目录
#         # 创建目录操作函数
#         os.makedirs(path)
#         print(path + ' 创建成功')
#         return True
#     else:
#         # 如果目录存在则不创建，并提示目录已存在
#         print(path + ' 目录已存在')
#         return False
#
#
# def read_csv(path_csv):
#     df = pd.read_csv(path_csv)
#     data = np.array(df)
#     return data
#
#
# def grid_name(data_dem,rows, cols):
#     # 获取有dem并且网格站点数>=1的网格编号
#     grid_row = []
#     grid_col = []
#     for row in range(rows):
#         for col in range(cols):
#             if data_dem[row][col] != -9999:
#             # if data_dem[row][col] != -9999:
#                 grid_row.append("{:03d}".format(row + 1))  # 要+1，变成行列的数字，而不是从0开始
#                 grid_col.append("{:03d}".format(col + 1))
#     return grid_row, grid_col
#
#
# def read_dir(path):
#     # 获取文件夹下所有txt文件名
#     filelist = os.listdir(path)
#     return filelist  # for filename in filelist
#
#
# def read_txt(file_name):
#     # 读取txt数据，以矩阵形式
#     file_txt = open(file_name)
#     lines = file_txt.readlines()
#
#     data = []
#     for line in lines:
#         data.append(line.strip('\n'))
#
#     data_ = []
#     for i in range(6, len(data)):
#         data_.append(data[i].strip(' ').split(' '))
#     data_ = np.array(data_)
#     data_ = data_.astype('float64')
#     return data_
#
#
# def txt_create(path_txt, num1, num2, val, name):
#     # 创建txt,并追加写入数据
#     file = open(path_txt + '\\' + name + "_{}{}.txt".format(num1, num2), 'a+')  # num1:list_row[],num2:list_col[]
#     file.write("{:.4f}".format(val) + '\n')
#
#
# if __name__ == '__main__':   #主程序开始
#     # path_dem1_csv = "G:\\shirun\\dem\\DEM.csv"
#     path_dem_csv = "G:\\shirun\\dem\\DEM.csv"
#
#     # data_s = read_csv(path_dem_csv)  #疑问：data_station = data_s[1:, 1:]
#     # data_dem1 = data_s[:,1:]
#     # # # print(data_station)
#
#     data_d = read_csv(path_dem_csv)
#     data_dem = data_d[:, 1:]
#     # print(data_dem)
#     # print(data_dem.shape)  # （44，41）
#     rows = data_dem.shape[0]
#     cols = data_dem.shape[1]
#
#     grid_row, grid_col = grid_name(data_dem,rows,cols)
#     print(grid_row,grid_col)
#
#     name = '12-2'  # cpc  early  final
#     path_p = 'G:\\shirun\\01_clip_data\\cpc4\\' + name
#     p_name = read_dir(path_p)
#     # print(p_name)
#
#     path_save = 'G:\\shirun\\02_final_data\\cpc4\\' + name
#     mkdir(path_save)
#     print(len(grid_row))
#
#     for i in range(len(grid_row)):
#         num1 = grid_row[i]  # str 02
#         num2 = grid_col[i]  # str 31
#         if os.path.exists(path_save + '\\' + name + '_' + num1 + num2 + '.txt'):
#             continue
#         else:
#             for p in p_name:
#                 file_name = path_p + '\\' + p
#                 data = read_txt(file_name)
#                 val = data[int(num1)-1, int(num2)-1]
#                 if val == -9999:
#                     print(file_name)
#                 else:
#                     txt_create(path_save, num1, num2, val, name)
#
# #############################################################################################################
#
# # import pandas as pd
# # import numpy as np
# # import math
# # import os
# # import csv
# # from collections import deque
# #
# #
# # def mkdir(path):
# #     # 去除首位空格
# #     path = path.strip()
# #     # 去除尾部 \ 符号
# #     path = path.rstrip("\\")
# #     # 判断路径是否存在
# #     # 存在     True
# #     # 不存在   False
# #     isExists = os.path.exists(path)
# #     # 判断结果
# #     if not isExists:
# #         # 如果不存在则创建目录
# #         # 创建目录操作函数
# #         os.makedirs(path)
# #         print(path + ' 创建成功')
# #         return True
# #     else:
# #         # 如果目录存在则不创建，并提示目录已存在
# #         print(path + ' 目录已存在')
# #         return False
# #
# #
# # def read_csv(path_csv):
# #     df = pd.read_csv(path_csv)
# #     data = np.array(df)
# #     return data
# #
# #
# # def grid_name(data_dem,rows, cols):
# #     # 获取有dem并且网格站点数>=1的网格编号
# #     grid_row = []
# #     grid_col = []
# #     for row in range(rows):
# #         for col in range(cols):
# #             if data_dem[row][col] != -9999:
# #             # if data_dem[row][col] != -9999:
# #                 grid_row.append("{:03d}".format(row + 1))  # 要+1，变成行列的数字，而不是从0开始
# #                 grid_col.append("{:03d}".format(col + 1))
# #     return grid_row, grid_col
# #
# #
# # def read_dir(path):
# #     # 获取文件夹下所有txt文件名
# #     filelist = os.listdir(path)
# #     return filelist  # for filename in filelist
# #
# #
# # def read_txt(file_name):
# #     # 读取txt数据，以矩阵形式
# #     file_txt = open(file_name)
# #     lines = file_txt.readlines()
# #
# #     data = []
# #     for line in lines:
# #         data.append(line.strip('\n'))
# #
# #     data_ = []
# #     for i in range(6, len(data)):
# #         data_.append(data[i].strip(' ').split(' '))
# #     data_ = np.array(data_)
# #     data_ = data_.astype('float64')
# #     return data_
# #
# #
# # def txt_create(path_txt, num1, num2, val, name):
# #     # 创建txt,并追加写入数据
# #     file = open(path_txt + '\\' + name + "_{}{}.txt".format(num1, num2), 'a+')  # num1:list_row[],num2:list_col[]
# #     file.write("{:.4f}".format(val) + '\n')
# #
# #
# # if __name__ == '__main__':   #主程序开始
# #     # path_dem1_csv = "G:\\shirun\\dem\\DEM.csv"
# #     path_dem_csv = "H:\\shirun\\dem\\DEM.csv"
# #
# #     # data_s = read_csv(path_dem_csv)  #疑问：data_station = data_s[1:, 1:]
# #     # data_dem1 = data_s[:,1:]
# #     # # # print(data_station)
# #
# #     data_d = read_csv(path_dem_csv)
# #     data_dem = data_d[:, 1:]
# #     # print(data_dem)
# #     # print(data_dem.shape)  # （44，41）
# #     rows = data_dem.shape[0]
# #     cols = data_dem.shape[1]
# #
# #     grid_row, grid_col = grid_name(data_dem,rows,cols)
# #     print(grid_row,grid_col)
# #     print(len(grid_row))
# #
# #     name = '6-8'
# #     path_p = 'H:\\shirun\\02_final_data\\cpc4\\' + name   # cpc  early  final
# #     p_name = read_dir(path_p)
# #
# #     list=[]
# #     for i in range(len(grid_row)):
# #         num1 = grid_row[i]  # str 02
# #         num2 = grid_col[i]  # str 31
# #         a = path_p + '\\' + name + '_' + num1 + num2 + '.txt'
# #         list.append(a)
# #         # print(list)
# #
# #     for p in p_name:
# #         file_name = path_p + '\\' + p
# #         # print(file_name)
# #         if file_name in list:
# #             continue
# #         else:
# #             os.remove(file_name)
# #     #
# #     #
