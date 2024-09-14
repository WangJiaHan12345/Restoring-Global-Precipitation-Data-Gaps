#
# # 原文链接https://zhuanlan.zhihu.com/p/59298710
#
# import numpy as np
# import os

# #得到文件夹内的所有文件名字
# def read_datas(file_dir):
#     data = []
#     for root, dirs, files in os.walk(file_dir):
#         data.append(files)
#     return data
#
#
#
# def read(filename):
#     """
#     Args:
#         filename: 文件名
#
#     Returns:
#         crain: 降水量mm <numpy.ndarray>
#         gsamp: 雨量计数量 <numpy.ndarray>
#     """
#     data = np.fromfile(filename, dtype="<f4")
#     data = data.reshape((880, 700), order='C')[::-1]
#     gsamp = data[:440]
#     crain = data[440:]
#     return crain, gsamp
#
#
# path = 'H:\\国家气象局\\2015\\2015\\'
# names = read_datas(path)[0]  # a_names得到的是一个文件夹中所有文件的所有名字
#
#
#
# for i in range(1,len(names),720):
#     filename = path + names[i]
#     crain, gsmap = read(filename)
#     # print(crain) # 降水量
#
#     output_path = 'H:\\青藏高原数据\\时间预测\\站点信息\\2016\\' + str(names[i]).split('-')[1].split('.')[0] + '.txt'
#
#     f = open(output_path,mode='w')
#     f.write('NCOLS        700\nNROWS        440\nXLLCORNER   70\nYLLCORNER    15\nCELLSIZE    0.1\nNODATA_VALUE   -999\n')
#
#     #写入数据
#     for m in range(len(gsmap)):
#         for n in range(len(gsmap[m])):
#             f.write(str(gsmap[m][n]))
#             f.write(' ')
#         f.write('\n')
#     f.close()



# 第二部分
# 24小时转为一天
import numpy as np
import os
import xlrd
import xlwt
import pandas as pd


#得到文件夹内的所有文件名字
def read_datas(file_dir):
    data = []
    for root, dirs, files in os.walk(file_dir):
        data.append(files)
    return data


path = 'H:\\中国区域数据\\国家气象局-0.1deg\\2016\\'
names = read_datas(path)[0]


for i in range(1,24,len(names)):
    data_day = np.zeros((440, 700)).astype('float64')
    for k in range(i+24):
        filename = path + names[k]
        f = open(filename,'r')
        data_hour = f.readlines()
        f.close()
        data_hour = np.array(data_hour)

        for m in range(440):
            for n in range(700):
                data_day[m][n] = data_day[m][n] + float(data_hour[m+6].split(' ')[n])

    print('1')

    output_path = 'H:\\中国区域数据\\国家气象局-0.1deg\\day\\' + str(names[i]).split('.')[0][:8] + '.txt'
    f1 = open(output_path, mode='w')
    f1.write('NCOLS        700\nNROWS        440\nXLLCORNER   70\nYLLCORNER    15\nCELLSIZE    0.1\nNODATA_VALUE   -9999\n')

    # 写入数据
    for a in range(len(data_day)):
        for b in range(len(data_day[a])):
            if data_day[a][b] < 0:
                data_day[a][b] = 0
            f1.write(str(data_day[a][b]))
            f1.write(' ')
        f1.write('\n')
    f1.close()