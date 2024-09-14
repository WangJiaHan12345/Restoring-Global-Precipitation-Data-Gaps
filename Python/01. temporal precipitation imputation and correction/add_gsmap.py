# ！/usr/bin/env python
# encoding: utf-8
###引入第三方模块###
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import tensorflow._api.v2.compat.v1 as tf
# tf.enable_eager_execution()
tf.disable_v2_behavior()
# tf.disable_eager_execution()
from numpy import *
from itertools import chain
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
from numpy.linalg import eig
from sklearn.preprocessing import StandardScaler

#注意力机制
from tensorflow.keras import backend as K
import math


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
                # data.append(line.strip('\n').strip())
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


###读取数据###
# 训练数据
earth = '南'
season = '12-2'
block = 'A'


# 测试数据
a_path_1 = 'H:\\时间预测\\四个区域可用数据\\ganhan\\03_class_data\\季节\\' + earth + '\\ceshi4\\early4\\' + season + '\\' + block + '\\'
# test_dem_path = 'H:\\时间预测\\四个区域可用数据\\shirun\\03_class_data\\季节\\' + earth + '\\ceshi_features4\\dem4\\' + season + '\\' + block + '\\'
test_lat_path = 'H:\\时间预测\\四个区域可用数据\\ganhan\\03_class_data\\季节\\' + earth + '\\ceshi_features4\\lat_wei4\\' + season + '\\' + block + '\\'
# test_lon_path = 'H:\\时间预测\\四个区域可用数据\\shirun\\03_class_data\\ceshi_features\\lon_jing\\' + block + '\\'
# test_slope_path = 'H:\\时间预测\\四个区域可用数据\\shirun\\03_class_data\\ceshi_features\\slope\\' + block + '\\'
# test_aspect_path = 'H:\\时间预测\\四个区域可用数据\\shirun\\03_class_data\\ceshi_features\\aspect\\' + block + '\\'
test_wendu_path = 'H:\\时间预测\\四个区域可用数据\\ganhan\\03_class_data\\季节\\' + earth + '\\ceshi_features4\\wendu4\\' + season + '\\' + block + '\\'
c_path_1 = 'H:\\时间预测\\四个区域可用数据\\ganhan\\03_class_data\\季节\\' + earth + '\\ceshi4\\cpc4\\' + season + '\\' + block + '\\'
d_path_1 = 'H:\\时间预测\\四个区域可用数据\\ganhan\\03_class_data\\季节\\' + earth + '\\ceshi4\\final4\\' + season + '\\' + block + '\\'
e_path_1 = 'H:\\时间预测\\四个区域可用数据\\ganhan\\03_class_data\\季节\\' + earth + '\\ceshi4\\mvk4\\' + season + '\\' + block + '\\'
f_path_1 = 'H:\\时间预测\\四个区域可用数据\\ganhan\\03_class_data\\季节\\' + earth + '\\ceshi4\\gauge4\\' + season + '\\' + block + '\\'


a_names_1 = read_datas(a_path_1)[0]  ##这里得到的是各个mvk_xx的名字
# test_dem_names = read_datas(test_dem_path)[0]
test_lat_names = read_datas(test_lat_path)[0]
# test_lon_names = read_datas(test_lon_path)[0]
# test_slope_names = read_datas(test_slope_path)[0]
# test_aspect_names = read_datas(test_aspect_path)[0]
test_wendu_names = read_datas(test_wendu_path)[0]
c_names_1 = read_datas(c_path_1)[0]
d_names_1 = read_datas(d_path_1)[0]
e_names_1 = read_datas(e_path_1)[0]
f_names_1 = read_datas(f_path_1)[0]




a_in_1 = []
# test_dem = []
test_lat = []
# test_lon = []
# test_slope = []
# test_aspect = []
test_wendu = []
c_in_1 = []
d_in_1 = []
e_in_1 = []
f_in_1 = []



a_in_1 = save_datas(a_path_1, a_names_1, a_in_1)
# test_dem = save_datas(test_dem_path, test_dem_names, test_dem)
test_lat = save_datas(test_lat_path, test_lat_names, test_lat)
# test_lon = save_datas(test_lon_path, test_lon_names, test_lon)
# test_slope = save_datas(test_slope_path, test_slope_names, test_slope)
# test_aspect = save_datas(test_aspect_path, test_aspect_names, test_aspect)
test_wendu = save_datas(test_wendu_path, test_wendu_names, test_wendu)
c_in_1 = save_datas(c_path_1, c_names_1, c_in_1)
d_in_1 = save_datas(d_path_1, d_names_1, d_in_1)
e_in_1 = save_datas(e_path_1, e_names_1, e_in_1)
f_in_1 = save_datas(f_path_1, f_names_1, f_in_1)



a_in_1 = np.array(a_in_1)
a_in_1 = a_in_1.astype('float32')
a_in_1 = list(chain.from_iterable(a_in_1))
a_in_1 = np.array([a_in_1]).T

# test_dem = np.array(test_dem)
# test_dem = test_dem.astype('float32')
# test_dem = list(chain.from_iterable(test_dem))
# test_dem = np.array([test_dem]).T

test_lat = np.array(test_lat)
test_lat = test_lat.astype('float32')
test_lat = list(chain.from_iterable(test_lat))
test_lat = np.array([test_lat]).T


# test_lon = np.array(test_lon)
# test_lon = test_lon.astype('float32')
# test_lon = list(chain.from_iterable(test_lon))
# test_lon = np.array([test_lon]).T


# test_slope = np.array(test_slope)
# test_slope = test_slope.astype('float32')
# test_slope = list(chain.from_iterable(test_slope))
# test_slope = np.array([test_slope]).T


# test_aspect = np.array(test_aspect)
# test_aspect = test_aspect.astype('float32')
# test_aspect = list(chain.from_iterable(test_aspect))
# test_aspect = np.array([test_aspect]).T

test_wendu = np.array(test_wendu)
test_wendu = test_wendu.astype('float32')
test_wendu = list(chain.from_iterable(test_wendu))
test_wendu = np.array([test_wendu]).T

c_in_1 = np.array(c_in_1)
c_in_1 = c_in_1.astype('float32')
c_in_1 = list(chain.from_iterable(c_in_1))
c_in_1 = np.array([c_in_1]).T

d_in_1 = np.array(d_in_1)
d_in_1 = d_in_1.astype('float32')
d_in_1 = list(chain.from_iterable(d_in_1))
d_in_1 = np.array([d_in_1]).T


e_in_1 = np.array(e_in_1)
e_in_1 = e_in_1.astype('float32')
e_in_1 = list(chain.from_iterable(e_in_1))
e_in_1 = np.array([e_in_1]).T


f_in_1 = np.array(f_in_1)
f_in_1 = f_in_1.astype('float32')
f_in_1 = list(chain.from_iterable(f_in_1))
f_in_1 = np.array([f_in_1]).T


grid = len(a_names_1)
day2 = 90   # 3-5:92  6-8:92  9-11:91  12-2:90
test_count = day2*grid #这里与一次性用几个格子有关 1096是一个格子 365*n个格子
test_data = np.zeros((test_count, 8))
test_data[:, [0]] = a_in_1[:, [0]]
# test_data[:, [1]] = test_dem[:, [0]]
test_data[:, [1]] = test_lat[:, [0]]
# test_data[:, [3]] = test_lon[:, [0]]
# test_data[:, [4]] = test_slope[:, [0]]
# test_data[:, [5]] = test_aspect[:, [0]]
test_data[:, [2]] = test_wendu[:, [0]]
test_data[:, [3]] = c_in_1[:, [0]]
# 用来存放a_in_1 和 d_in_1
test_data[:, [4]] = a_in_1[:, [0]]
test_data[:, [5]] = d_in_1[:, [0]]
test_data[:, [6]] = e_in_1[:, [0]]
test_data[:, [7]] = f_in_1[:, [0]]



##############################################################################################
###定义设置LSTM常量###
rnn_unit = 60
# 隐层单元的数量  100适用于小的
input_size = 3 # 输入矩阵维度
output_size = 1  # 输出矩阵维度
lr = 0.00001 # 学习率

batch_size=60
###########################
###########################
#一定要记得改
time_step=29 # 3-5 6-8:23  9-11: 30  12-2:29

# 12-2月的数据
def get_test_data(time_step=time_step):
    mean = np.mean(test_data, axis=0)
    std = np.std(test_data, axis=0)

    normalized_test_data = (test_data - mean) / std

    # num代表网格的个数
    size1 = 59 // time_step  # 有size个sample   '//'相当于python中的'/'
    size2 = 31 // time_step  # 有size个sample   '//'相当于python中的'/'
    test_x, test_y = [], []
    early_data, final_data = [], []
    mvk_data,gauge_data = [],[]

    for num in range(len(normalized_test_data) // 90):  #90代表12-2的天数
        for i in range(size1): #73
            w = num*90
            x = normalized_test_data[w + i* time_step:w + (i+1) * time_step, :3]
            y = normalized_test_data[w + i* time_step:w + (i+1) * time_step, 3:4]
            m = test_data[w + i * time_step:w + (i + 1) * time_step, 4:5]
            n = test_data[w + i * time_step:w + (i + 1) * time_step, 5:6]
            p = test_data[w + i * time_step:w + (i + 1) * time_step, 6:7]
            q = test_data[w + i * time_step:w + (i + 1) * time_step, 7:]
            early_data.extend(m)
            final_data.extend(n)
            mvk_data.extend(p)
            gauge_data.extend(q)
            test_x.append(x.tolist())
            test_y.extend(y)

        for i in range(size2):
            w = num*90 + 59
            x = normalized_test_data[w + i * time_step: w + (i + 1) * time_step, :3]
            y = normalized_test_data[w + i * time_step: w + (i + 1) * time_step, 3:4]
            m = test_data[w + i * time_step:w + (i + 1) * time_step, 4:5]
            n = test_data[w + i * time_step:w + (i + 1) * time_step, 5:6]
            p = test_data[w + i * time_step:w + (i + 1) * time_step, 6:7]
            q = test_data[w + i * time_step:w + (i + 1) * time_step, 7:]
            early_data.extend(m)
            final_data.extend(n)
            mvk_data.extend(p)
            gauge_data.extend(q)
            test_x.append(x.tolist())
            test_y.extend(y)

    return mean, std, test_x, test_y, early_data, final_data,mvk_data,gauge_data

# 3-5月 6-8月的数据
# def get_test_data(time_step=time_step):
#     mean = np.mean(test_data, axis=0)
#     std = np.std(test_data, axis=0)
#
#     normalized_test_data = (test_data - mean) / std
#
#     # num代表网格的个数
#     size = 92 // time_step  # 有size个sample   '//'相当于python中的'/'
#     test_x, test_y = [], []
#     early_data, final_data = [], []
#     mvk_data,gauge_data = [],[]
#
#     for num in range(len(normalized_test_data) // 92):  #90代表12-2的天数
#         for i in range(size):
#             w = num*91
#             x = normalized_test_data[w + i* time_step:w + (i+1) * time_step, :3]
#             y = normalized_test_data[w + i* time_step:w + (i+1) * time_step, 3:4]
#             m = test_data[w + i * time_step:w + (i + 1) * time_step, 4:5]
#             n = test_data[w + i * time_step:w + (i + 1) * time_step, 5:6]
#             p = test_data[w + i * time_step:w + (i + 1) * time_step, 6:7]
#             q = test_data[w + i * time_step:w + (i + 1) * time_step, 7:]
#             test_x.append(x.tolist())
#             test_y.extend(y)
#             early_data.extend(m)
#             final_data.extend(n)
#             mvk_data.extend(p)
#             gauge_data.extend(q)
#
#     return mean, std, test_x, test_y, early_data, final_data,mvk_data,gauge_data


#9-11月的数据
# def get_test_data(time_step = time_step):
#     mean = np.mean(test_data, axis=0)
#     std = np.std(test_data, axis=0)
#
#     normalized_test_data = (test_data - mean) / std  # 标准化
#
#     # num代表网格的个数
#     size = 91 // time_step  # 有size个sample   '//'相当于python中的'/'
#     test_x, test_y = [], []
#     early_data, final_data = [], []
#     mvk_data,gauge_data = [], []
#
#     for num in range(len(normalized_test_data) // 91):  #90代表12-2的天数
#         for i in range(size):
#             w = num*91
#             x = normalized_test_data[w + i* time_step:w + (i+1) * time_step, :3]
#             y = normalized_test_data[w + i* time_step:w + (i+1) * time_step, 3:4]
#             m = test_data[w + i * time_step:w + (i + 1) * time_step, 4:5]
#             n = test_data[w + i * time_step:w + (i + 1) * time_step, 5:6]
#             p = test_data[w + i * time_step:w + (i + 1) * time_step, 6:7]
#             q = test_data[w + i * time_step:w + (i + 1) * time_step, 7:]
#             test_x.append(x.tolist())
#             test_y.extend(y)
#             early_data.extend(m)
#             final_data.extend(n)
#             mvk_data.extend(p)
#             gauge_data.extend(q)
#
#     return mean, std, test_x, test_y, early_data, final_data,mvk_data,gauge_data


path_random_save = 'H:\\时间预测\\四个区域可用数据\\ganhan\\result\\季节\\' + earth + '\\' + season + '\\'
mkdir(path_random_save)  # 创建一个目录存放结果

# path_early_save = path_random_save + 'early\\'
# mkdir(path_early_save)
#
# path_final_save = path_random_save + 'final\\'
# mkdir(path_final_save)
#
# path_cpc_save = path_random_save + 'cpc\\'
# mkdir(path_cpc_save)

path_mvk_save = path_random_save + 'mvk\\'
mkdir(path_mvk_save)

path_gauge_save = path_random_save + 'gauge\\'
mkdir(path_gauge_save)


mean, std, test_x, test_y, early_data, final_data, mvk_data, gauge_data = get_test_data(time_step)
test_y = np.array(test_y) * std[3] + mean[3]

# with open(path_early_save + block +'_data.txt', 'w')as f:
#     for j in range(len(early_data)):
#         f.writelines(str(early_data[j][0]))
#         f.writelines('\n')
#
#
# with open(path_final_save + block +'_data.txt', 'w')as f:
#     for j in range(len(final_data)):
#         f.writelines(str(final_data[j][0]))
#         f.writelines('\n')
#
# with open(path_cpc_save + block +'_data.txt', 'w')as f:
#     for j in range(len(test_y)):
#         f.writelines(str(test_y[j][0]))
#         f.writelines('\n')


with open(path_mvk_save + block +'_data.txt', 'w')as f:
    for j in range(len(mvk_data)):
        f.writelines(str(mvk_data[j][0]))
        f.writelines('\n')

with open(path_gauge_save + block +'_data.txt', 'w')as f:
    for j in range(len(gauge_data)):
        f.writelines(str(gauge_data[j][0]))
        f.writelines('\n')

