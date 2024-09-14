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


#自注意力机制
def SE_Block(input_tensor,ratio = 16):
    input_shape = K.int_shape(input_tensor)
    #GlobalAveragePooling1D
    squeeze = tf.keras.layers.GlobalAveragePooling1D()(input_tensor)
    excitation = tf.reshape(squeeze, [-1, 1, input_shape[-1]])
    excitation = tf.keras.layers.Dense(units = input_shape[-1]//ratio, kernel_initializer='he_normal',activation='relu')(excitation)
    excitation = tf.keras.layers.Dense(units = input_shape[-1],activation='sigmoid')(excitation)
    scale = tf.keras.layers.Multiply()([input_tensor, excitation])
    return scale


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


def rmse_count(x_in, s_in,num):
    # 求均方根误差
    sum = 0
    cha = 0
    for i in range(num):
            cha = (x_in[i][0] - s_in[i][0])
            sum += pow(cha, 2)
    rmse = math.sqrt(sum / num)
    return rmse


def cc_count(x_in, s_in,num):
    # 求相关系数
    average_x = np.mean(x_in)
    average_s = np.mean(s_in)
    shang = 0
    xia_1 = 0
    xia_2 = 0
    for i in range(num):
            shang += (x_in[i][0] - average_x) * (s_in[i][0]- average_s)
            xia_1 += pow((x_in[i][0] - average_x), 2)
            xia_2 += pow((s_in[i][0] - average_s), 2)
    cc = shang / (math.sqrt(xia_1) * math.sqrt(xia_2))
    return cc


def mae_count(x_in, s_in,num):
    sum = 0
    cha = 0
    for i in range(num):
        cha = (x_in[i][0] - s_in[i][0])
        sum += abs(cha)
    mae = sum / num
    return mae

def me_count(x_in, s_in,num):
    sum = 0
    cha = 0
    for i in range(num):
        cha = (x_in[i][0] - s_in[i][0])
        sum += cha
    me = sum / num
    return me


def bias_count(x_in, s_in,num):
    # 求相对偏差
    shang = 0
    xia = np.sum(x_in)
    for i in range(num):
            shang += x_in[i][0]- s_in[i][0]
    bias = (shang / xia) * 100
    return bias

def abias_count(x_in, s_in,num):
    # 求绝对偏差
    shang = 0
    xia = np.sum(s_in)
    for j in range(num):
        shang += abs(x_in[j][0]-s_in[j][0])
    abias = (shang / xia) * 100
    return abias


def pod(x_in, s_in,num):
    # 求pod
    n10 = 0
    n01 = 0
    n11 = 0
    for j in range(num):
        # CPC
            if s_in[j][0] >= 2.4 and x_in[j][0] < 2.4:
                n01 = n01 + 1
            # 卫星
            if x_in[j][0] >= 2.4 and s_in[j][0]< 2.4:
                n10 = n10 + 1
            #     cpc+卫星
            if x_in[j][0]>= 2.4 and s_in[j][0]>= 2.4:
                n11 = n11 +1
    temp = n11 + n01
    if temp == 0:
        temp = 1
    pod = n11/ temp
    return pod

def far(x_in, s_in,num):
    # 求pod
    n10 = 0
    n01 = 0
    n11 = 0
    for j in range(num):
            #  CPC
            if s_in[j][0] >= 2.4 and x_in[j][0] < 2.4:
                n01 = n01 + 1
            # 卫星
            if x_in[j][0] >= 2.4 and s_in[j][0] < 2.4:
                n10 = n10 + 1
            # cpc+卫星
            if x_in[j][0] >= 2.4 and s_in[j][0]>= 2.4:
                n11 = n11 + 1
    temp = n11 + n10
    if temp == 0:
        temp = 1
    far = n10/ temp
    return far

def csi(x_in, s_in,num):
    # 求pod
    n10 = 0
    n01 = 0
    n11 = 0
    for j in range(num):
            # CPC
            if s_in[j][0] >= 2.4 and x_in[j][0] < 2.4:
                n01 = n01 + 1
            # 卫星
            if x_in[j][0] >= 2.4 and s_in[j][0] < 2.4:
                n10 = n10 + 1
            #     cpc+卫星
            if x_in[j][0] >= 2.4 and s_in[j][0] >= 2.4:
                n11 = n11 + 1
    temp = n11+n10+n01
    if temp == 0:
        temp = 1
    csi = n11/ temp
    return csi


climates = ['随意']
for climate in climates:
    #要不要加上坡度和坡向还有高程
    earths = ['北','南']
    for earth in earths:
        seasons = ['3-5','6-8','9-11','12-2']
        for season in seasons:
            a_path = 'H:\\时间预测\\四个区域可用数据\\只选用地面站点final\\02_grid_ns_data\\xunlian4\\'+ earth +'\\Final\\'+season+'\\'
            c_path = 'H:\\时间预测\\四个区域可用数据\\只选用地面站点final\\02_grid_ns_data\\xunlian4\\'+ earth +'\\cpc\\'+season+'\\'
            d_path = 'H:\\时间预测\\四个区域可用数据\\只选用地面站点final\\02_grid_ns_data\\xunlian4\\'+ earth +'\\Early\\'+season+'\\'


            train = True

            a_names = read_datas(a_path)[0]  # a_names得到的是一个文件夹中所有文件的所有名字
            c_names = read_datas(c_path)[0]
            d_names = read_datas(d_path)[0]

            a_in = []
            c_in = []
            d_in = []


            a_in = save_datas(a_path, a_names, a_in)   # a_in的形式是二维数组，每一维得到的是一个各自的时间序列数据
            c_in = save_datas(c_path, c_names, c_in)
            d_in = save_datas(d_path, d_names, d_in)



            path_random_save = 'H:\\时间预测\\四个区域可用数据\\只选用地面站点final\\result(优化lstm+不分块+不加因子+训练次数变多了)\\' + earth + '\\' + season + '\\'
            mkdir(path_random_save)  # 创建一个目录存放结果


            a_in = np.array(a_in)
            a_in = a_in.astype('float32')
            a_in = list(chain.from_iterable(a_in))
            a_in = np.array([a_in]).T
            # print('a_in',a_in)

            c_in = np.array(c_in)
            c_in = c_in.astype('float32')
            c_in = list(chain.from_iterable(c_in))
            c_in = np.array([c_in]).T


            d_in = np.array(d_in)
            d_in = d_in.astype('float32')
            d_in = list(chain.from_iterable(d_in))
            d_in = np.array([d_in]).T


            grid = len(a_names)

            if season == '3-5':
                day = 276
            elif season == '6-8':
                day = 276
            elif season == '9-11':
                day = 273
            else:
                day = 271


            train_count = day*grid
            train_data = np.zeros((train_count,2))
            train_data[:, [0]] = a_in[:, [0]]
            train_data[:, [1]] = c_in[:, [0]]


            # 测试数据
            a_path_1 = 'H:\\时间预测\\四个区域可用数据\\只选用地面站点final\\02_grid_ns_data\\ceshi4\\' + earth + '\\Final\\' + season + '\\'
            c_path_1 = 'H:\\时间预测\\四个区域可用数据\\只选用地面站点final\\02_grid_ns_data\\ceshi4\\' + earth + '\\cpc\\' + season + '\\'
            d_path_1 = 'H:\\时间预测\\四个区域可用数据\\只选用地面站点final\\02_grid_ns_data\\ceshi4\\' + earth + '\\Early\\' + season + '\\'

            a_names_1 = read_datas(a_path_1)[0]  ##这里得到的是各个mvk_xx的名字
            c_names_1 = read_datas(c_path_1)[0]
            d_names_1 = read_datas(d_path_1)[0]

            a_in_1 = []
            c_in_1 = []
            d_in_1 = []

            a_in_1 = save_datas(a_path_1, a_names_1, a_in_1)
            c_in_1 = save_datas(c_path_1, c_names_1, c_in_1)
            d_in_1 = save_datas(d_path_1, d_names_1, d_in_1)

            a_in_1 = np.array(a_in_1)
            a_in_1 = a_in_1.astype('float32')
            a_in_1 = list(chain.from_iterable(a_in_1))
            a_in_1 = np.array([a_in_1]).T


            c_in_1 = np.array(c_in_1)
            c_in_1 = c_in_1.astype('float32')
            c_in_1 = list(chain.from_iterable(c_in_1))
            c_in_1 = np.array([c_in_1]).T

            d_in_1 = np.array(d_in_1)
            d_in_1 = d_in_1.astype('float32')
            d_in_1 = list(chain.from_iterable(d_in_1))
            d_in_1 = np.array([d_in_1]).T

            if season == '3-5':
                day2 = 92
            elif season == '6-8':
                day2 = 92
            elif season == '9-11':
                day2 = 91
            else:
                day2 = 90

            test_count = day2*grid
            test_data = np.zeros((test_count, 4))
            test_data[:, [0]] = a_in_1[:, [0]]
            test_data[:, [1]] = c_in_1[:, [0]]
            # 用来存放a_in_1 和 d_in_1
            test_data[:, [2]] = a_in_1[:, [0]]
            test_data[:, [3]] = d_in_1[:, [0]]



            ##############################################################################################
            ###定义设置LSTM常量###
            rnn_unit = 8
            # 隐层单元的数量  100适用于小的
            input_size = 1 # 输入矩阵维度
            output_size = 1  # 输出矩阵维度
            lr = 0.00001 # 学习率

            batch_size= 64
            ###########################
            ###########################
            #一定要记得改

            if season == '3-5':
                time_step = 10
            elif season == '6-8':
                time_step = 10
            elif season == '9-11':
                time_step = 10
            else:
                time_step = 5


            # 12-2月的数据
            def get_train_data_12_2(batch_size=batch_size, time_step=time_step):  # 90是按照格网数来的  或者将5改为73
                batch_index = []
                ########################################################################################################
                normalized_train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)
                train_x, train_y = [], []  # 训练集 train_x是需要训练的数据  train_y是需要和结果比较的数据

                # 得到所有数据
                for i in range(len(normalized_train_data) - time_step):
                    x = normalized_train_data[i:i + time_step, :1]
                    y = normalized_train_data[i:i + time_step, 1:]
                    train_x.append(x.tolist())
                    train_y.append(y.tolist())

                for num in range(len(normalized_train_data) // 271):  # 271代表总的天数
                    for i in range(59 - time_step):
                        w = i + num * 271
                        if i % batch_size == 0:
                            batch_index.append(w)
                    batch_index.append((271 * num + 59 - time_step))

                    for i in range(59, 150 - time_step):
                        w = i + num * 271
                        if (i - 59) % batch_size == 0:
                            batch_index.append(w)
                    batch_index.append((271 * num + 150 - time_step))

                    for i in range(150, 240 - time_step):
                        w = i + num * 271
                        if (i - 150) % batch_size == 0:
                            batch_index.append(w)
                    batch_index.append((271 * num + 240 - time_step))

                    for i in range(240, 271 - time_step):
                        w = i + num * 271
                        if (i - 240) % batch_size == 0:
                            batch_index.append(w)
                    batch_index.append((271 * (num + 1) - time_step))

                return batch_index, train_x, train_y


            def get_test_data_12_2(time_step=time_step):
                mean = np.mean(test_data, axis=0)
                std = np.std(test_data, axis=0)

                normalized_test_data = (test_data - mean) / std

                # num代表网格的个数
                size1 = 59 // time_step  # 有size个sample   '//'相当于python中的'/'
                size2 = 31 // time_step  # 有size个sample   '//'相当于python中的'/'
                test_x, test_y = [], []
                early_data, final_data = [], []

                for num in range(len(normalized_test_data) // 90):  # 90代表12-2的天数
                    for i in range(size1):  # 73
                        w = num * 90
                        x = normalized_test_data[w + i * time_step:w + (i + 1) * time_step, :1]
                        y = normalized_test_data[w + i * time_step:w + (i + 1) * time_step, 1:2]
                        m = test_data[w + i * time_step:w + (i + 1) * time_step, 2:3]
                        n = test_data[w + i * time_step:w + (i + 1) * time_step, 3:4]
                        final_data.extend(m)
                        early_data.extend(n)
                        test_x.append(x.tolist())
                        test_y.extend(y)

                    for i in range(size2):
                        w = num * 90 + 59
                        x = normalized_test_data[w + i * time_step: w + (i + 1) * time_step, :1]
                        y = normalized_test_data[w + i * time_step: w + (i + 1) * time_step, 1:2]
                        m = test_data[w + i * time_step:w + (i + 1) * time_step, 2:3]
                        n = test_data[w + i * time_step:w + (i + 1) * time_step, 3:4]
                        final_data.extend(m)
                        early_data.extend(n)
                        test_x.append(x.tolist())
                        test_y.extend(y)

                return mean, std, test_x, test_y, early_data, final_data


            # 3-5月 6-8月的数据
            def get_train_data_3_8(batch_size=batch_size, time_step=time_step):  # 90是按照格网数来的  或者将5改为73
                batch_index = []
                ########################################################################################################
                normalized_train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)
                train_x, train_y = [], []  # 训练集 train_x是需要训练的数据  train_y是需要和结果比较的数据

                # 得到所有数据
                for i in range(len(normalized_train_data) - time_step):
                    x = normalized_train_data[i:i + time_step, :1]
                    y = normalized_train_data[i:i + time_step, 1:]
                    train_x.append(x.tolist())
                    train_y.append(y.tolist())

                for num in range(len(normalized_train_data) // 276):  # 271代表总的天数
                    for i in range(92 - time_step):
                        w = i + num * 276
                        if i % batch_size == 0:
                            batch_index.append(w)
                    batch_index.append((276 * num + 92 - time_step))

                    for i in range(92, 184 - time_step):
                        w = i + num * 276
                        if (i - 92) % batch_size == 0:
                            batch_index.append(w)
                    batch_index.append((276 * num + 184 - time_step))

                    for i in range(184, 276 - time_step):
                        w = i + num * 276
                        if (i - 184) % batch_size == 0:
                            batch_index.append(w)
                    batch_index.append((276 * num + 276 - time_step))

                return batch_index, train_x, train_y


            def get_test_data_3_8(time_step=time_step):
                mean = np.mean(test_data, axis=0)
                std = np.std(test_data, axis=0)

                normalized_test_data = (test_data - mean) / std

                # num代表网格的个数
                size = 92 // time_step  # 有size个sample   '//'相当于python中的'/'
                test_x, test_y = [], []
                early_data, final_data = [], []

                for num in range(len(normalized_test_data) // 92):
                    for i in range(size):
                        w = num * 92
                        x = normalized_test_data[w + i * time_step:w + (i + 1) * time_step, :1]
                        y = normalized_test_data[w + i * time_step:w + (i + 1) * time_step, 1:2]
                        m = test_data[w + i * time_step:w + (i + 1) * time_step, 2:3]
                        n = test_data[w + i * time_step:w + (i + 1) * time_step, 3:4]
                        test_x.append(x.tolist())
                        test_y.extend(y)
                        final_data.extend(m)
                        early_data.extend(n)

                return mean, std, test_x, test_y, early_data, final_data


            # 9-11月的数据
            def get_train_data_9_11(batch_size=batch_size, time_step=time_step):  # 90是按照格网数来的  或者将5改为73
                batch_index = []
                ########################################################################################################
                normalized_train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)
                train_x, train_y = [], []  # 训练集 train_x是需要训练的数据  train_y是需要和结果比较的数据

                # 得到所有数据
                for i in range(len(normalized_train_data) - time_step):
                    x = normalized_train_data[i:i + time_step, :1]
                    y = normalized_train_data[i:i + time_step, 1:]
                    train_x.append(x.tolist())
                    train_y.append(y.tolist())

                for num in range(len(normalized_train_data) // 273):  # 271代表总的天数
                    for i in range(91 - time_step):
                        w = i + num * 273
                        if i % batch_size == 0:
                            batch_index.append(w)
                    batch_index.append((273 * num + 91 - time_step))

                    for i in range(91, 182 - time_step):
                        w = i + num * 273
                        if (i - 91) % batch_size == 0:
                            batch_index.append(w)
                    batch_index.append((273 * num + 182 - time_step))

                    for i in range(182, 273 - time_step):
                        w = i + num * 273
                        if (i - 182) % batch_size == 0:
                            batch_index.append(w)
                    batch_index.append((273 * num + 273 - time_step))

                return batch_index, train_x, train_y


            def get_test_data_9_11(time_step=time_step):
                mean = np.mean(test_data, axis=0)
                std = np.std(test_data, axis=0)

                normalized_test_data = (test_data - mean) / std  # 标准化

                # num代表网格的个数
                size = 91 // time_step  # 有size个sample   '//'相当于python中的'/'
                test_x, test_y = [], []
                early_data, final_data = [], []

                for num in range(len(normalized_test_data) // 91):  # 90代表12-2的天数
                    for i in range(size):
                        w = num * 91
                        x = normalized_test_data[w + i * time_step:w + (i + 1) * time_step, :1]
                        y = normalized_test_data[w + i * time_step:w + (i + 1) * time_step, 1:2]
                        m = test_data[w + i * time_step:w + (i + 1) * time_step, 2:3]
                        n = test_data[w + i * time_step:w + (i + 1) * time_step, 3:4]
                        test_x.append(x.tolist())
                        test_y.extend(y)
                        final_data.extend(m)
                        early_data.extend(n)

                return mean, std, test_x, test_y, early_data, final_data


            # ——————————————————定义LSTM网络权重和偏置——————————————————
            # 输入层、输出层权重、偏置

            weights = {
                'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
                'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
            }

            biases = {
                'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
                'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
            }

            # ——————————————————定义LSTM网络——————————————————
            # 二层
            def lstm(X):
                batch_size=tf.shape(X)[0]
                time_step=tf.shape(X)[1]
                w_in=weights['in']
                b_in=biases['in']

                input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
                input_rnn=tf.matmul(input,w_in)+b_in
                input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入

                cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit, activation=tf.nn.relu)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.8)

                # 二层
                # cell1 = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit, activation=tf.nn.relu)
                # cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.8)
                # cell2 = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit,activation=tf.nn.relu)
                # cell2 = tf.nn.rnn_cell.DropoutWrapper(cell2, output_keep_prob=0.8)
                # cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell1, cell2],state_is_tuple=True)
                # cell = cell1

                init_state=cell.zero_state(batch_size,dtype=tf.float32)
                output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果


                output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入

                w_out=weights['out']
                b_out=biases['out']
                pred=tf.matmul(output,w_out)+b_out
                return pred,final_states


            # ——————————————————LSTM模型训练—————————————————

            path_train_save = path_random_save + 'modelnew\\'
            mkdir(path_train_save)

            ###############将预测的结果存放起来##################################
            path_predict_save = path_random_save + 'ANN\\'
            mkdir(path_predict_save)

            path_early_save = path_random_save + 'early\\'
            mkdir(path_early_save)


            path_final_save = path_random_save + 'final\\'
            mkdir(path_final_save)


            path_cpc_save = path_random_save + 'cpc\\'
            mkdir(path_cpc_save)

            #——————————————————训练模型——————————————————
            def train_lstm(batch_size=batch_size,time_step=time_step):   #将time_step改为73
                X = tf.placeholder(tf.float32, shape=[None,time_step,input_size])
                Y = tf.placeholder(tf.float32, shape=[None,time_step,output_size])

                if season == '3-5':
                    batch_index, train_x, train_y = get_train_data_3_8(batch_size, time_step)
                elif season == '6-8':
                    batch_index, train_x, train_y = get_train_data_3_8(batch_size, time_step)
                elif season == '9-11':
                    batch_index, train_x, train_y = get_train_data_9_11(batch_size, time_step)
                else:
                    batch_index, train_x, train_y = get_train_data_12_2(batch_size, time_step)

                pred,_=lstm(X)

                #损失函数
                loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))

                with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)


                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())


                    for i in range(21):
                        for step in range(len(batch_index)-1):

                            _, loss_ = sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})

                        if i % 2 == 0:
                            print(i, loss_)
                            print("保存模型：",saver.save(sess,path_train_save+'model.ckpt',global_step=i))



            # ————————————————LSTM模型预测————————————————————

            def test_lstm(time_step=time_step):
                X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])  # 创建输入流图

                if season == '3-5':
                    mean, std, test_x, test_y, early_data, final_data = get_test_data_3_8(time_step)
                elif season == '6-8':
                    mean, std, test_x, test_y, early_data, final_data = get_test_data_3_8(time_step)
                elif season == '9-11':
                    mean, std, test_x, test_y, early_data, final_data = get_test_data_9_11(time_step)
                else:
                    mean, std, test_x, test_y, early_data, final_data = get_test_data_12_2(time_step)

                pred, _ = lstm(X)

                saver = tf.train.Saver(tf.global_variables())

                with tf.Session() as sess:
                    ###参数恢复，调用已经训练好的模型###
                    module_file = tf.train.latest_checkpoint(path_train_save)
                    saver.restore(sess, module_file)
                    test_predict = []


                    for step in range(len(test_x)):

                        prob = sess.run(pred, feed_dict={X: [test_x[step]]})
                        predict = prob.reshape(-1, 1)
                        test_predict.extend(predict)


                    #返标准化  这里也要改变维度
                    test_y = np.array(test_y) * std[1] + mean[1]
                    test_predict = np.array(test_predict) * std[1] + mean[1]


                    # 这里要知道a_names到底代表什么   以及搞清楚输入到神经网络里面的数据格式到底是什么样的
                    with open(path_predict_save +'data.txt', 'w')as f:
                        for j in range(len(test_predict)):
                            if test_predict[j][0] < 0:
                                test_predict[j][0] = 0
                            f.writelines(str(test_predict[j][0]))
                            f.writelines('\n')

                    with open(path_early_save +'data.txt', 'w')as f:
                        for j in range(len(early_data)):
                            f.writelines(str(early_data[j][0]))
                            f.writelines('\n')


                    with open(path_final_save +'data.txt', 'w')as f:
                        for j in range(len(final_data)):
                            f.writelines(str(final_data[j][0]))
                            f.writelines('\n')

                    with open(path_cpc_save +'data.txt', 'w')as f:
                        for j in range(len(test_y)):
                            f.writelines(str(test_y[j][0]))
                            f.writelines('\n')

                    num1 = len(test_predict)

                    print(climate)
                    CC_early = cc_count(early_data, test_y,num1)
                    print("CC_early :", CC_early)
                    BIAS_early = bias_count(early_data, test_y,num1)
                    print("BIAS_early :", BIAS_early)
                    RMSE_early = rmse_count(early_data, test_y,num1)
                    print("RMSE_early :", RMSE_early)
                    MAE_early = mae_count(early_data, test_y,num1)
                    print("MAE_early :", MAE_early)
                    ME_early = me_count(early_data, test_y,num1)
                    print("ME_early :", ME_early)
                    ABIAS_early = abias_count(early_data, test_y,num1)
                    print("ABIAS_early :", ABIAS_early)
                    POD_early = pod(early_data, test_y,num1)
                    print("POD_early :", POD_early)
                    FAR_early = far(early_data, test_y,num1)
                    print("FAR_early :", FAR_early)
                    CSI_early = csi(early_data, test_y,num1)
                    print("CSI_early :", CSI_early)
                    print('*' * 10)

                    CC_pre = cc_count(test_predict, test_y,num1)
                    print("CC_pre:", CC_pre)
                    BIAS_pre = bias_count(test_predict, test_y,num1)
                    print("BIAS_pre:", BIAS_pre)
                    RMSE_pre = rmse_count(test_predict, test_y,num1)
                    print("RMSE_pre:", RMSE_pre)
                    MAE_pre = mae_count(test_predict, test_y,num1)
                    print("MAE_pre:", MAE_pre)
                    ME_pre = me_count(test_predict, test_y,num1)
                    print("ME_pre:", ME_pre)
                    ABIAS_pre = abias_count(test_predict, test_y,num1)
                    print("ABIAS_pre:", ABIAS_pre)
                    POD_pre = pod(test_predict, test_y,num1)
                    print("POD_pre:", POD_pre)
                    FAR_pre = far(test_predict, test_y,num1)
                    print("FAR_pre:", FAR_pre)
                    CSI_pre = csi(test_predict, test_y,num1)
                    print("CSI_pre:", CSI_pre)
                    print('*' * 10)

                    CC_final = cc_count(final_data,test_y,num1)
                    print("CC_final:", CC_final)
                    BIAS_final = bias_count(final_data, test_y,num1)
                    print("BIAS_final:", BIAS_final)
                    RMSE_final = rmse_count(final_data, test_y,num1)
                    print("RMSE_final:", RMSE_final)
                    MAE_final = mae_count(final_data, test_y,num1)
                    print("MAE_final:", MAE_final)
                    ME_final = me_count(final_data, test_y,num1)
                    print("ME_final:", ME_final)
                    ABIAS_final = abias_count(final_data, test_y,num1)
                    print("ABIAS_final:", ABIAS_final)
                    POD_final = pod(final_data, test_y,num1)
                    print("POD_final:", POD_final)
                    FAR_final = far(final_data,test_y,num1)
                    print("FAR_final:", FAR_final)
                    CSI_final = csi(final_data, test_y,num1)
                    print("CSI_final:", CSI_final)


            train_lstm()
            print("开始预测")
            tf.compat.v1.get_variable_scope().reuse_variables()
            test_lstm()