# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:34:02 2019

@author: Gu
"""

import math
import tensorflow as tf
import os
# import cv2
import numpy as np
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import csv
from sklearn import metrics
import pandas as pd


def read_datas(file_dir):
    data = []   
    for root, dirs, files in os.walk(file_dir):
        data.append(files)
    return data 


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


def rmse_count(x_in, s_in, number1, number2):
    # 求均方根误差
    sum = 0
    count = 0
    for i in range(number1):
        for j in range(number2):
            count += 1
            cha = (x_in[i][j] - s_in[i][j])
            sum += pow(cha, 2)
    rmse = math.sqrt(sum / count)
    return rmse


def me_count(x_in, s_in, number1, number2):
    # 求平均误差
    sum = 0
    count = 0
    for i in range(number1):
        for j in range(number2):
            count += 1
            cha = (x_in[i][j] - s_in[i][j])
            sum += cha
    me = sum / count
    return me


def cc_count(x_in, s_in, number1, number2):
    # 求相关系数
    average_x = np.mean(x_in)
    average_s = np.mean(s_in)
    shang = 0
    xia_1 = 0
    xia_2 = 0
    for i in range(number1):
        for j in range(number2):
            shang += (x_in[i][j] - average_x) * (s_in[i][j] - average_s)
            xia_1 += pow((x_in[i][j] - average_x), 2)
            xia_2 += pow((s_in[i][j] - average_s), 2)
    cc = shang / (math.sqrt(xia_1) * math.sqrt(xia_2))
    return cc


def bias_count(x_in, s_in, number1, number2):
    # 求相对偏差
    shang = 0
    xia = np.sum(x_in)
    for i in range(number1):
        for j in range(number2):
            shang += s_in[i][j] - x_in[i][j]
    bias = (shang / xia) * 100
    return bias


# a_path = 'E:/LinJN_bishe/暴雨数据/上中下游/下/gsmap/'
# b_path = 'E:/LinJN_bishe/暴雨数据/上中下游/下/station/'
# c_path = 'E:/LinJN_bishe/暴雨数据/上中下游/下/standard/'
# a_path = 'E:/LinJN_bishe/暴雨数据/不同海拔数据/C/gsmap_gauge/'
# b_path = 'E:/LinJN_bishe/暴雨数据/不同海拔数据/C/station/'
# c_path = 'E:/LinJN_bishe/暴雨数据/不同海拔数据/C/standard/'
#==============================================================================
a_path = 'F:/水文资料/各种数据/短时暴雨序列/00_DEM_Rain/A/gsmap_mvk/'
# b_path = 'E:/LinJN_bishe/暴雨数据/00_DEM_Rain/A/station/'
c_path = 'F:/水文资料/各种数据/短时暴雨序列/00_DEM_Rain/A/standard/'
d_path = 'F:/水文资料/各种数据/短时暴雨序列/00_DEM_Rain/A/gsmap_gauge/'

# a_path = 'E:/LinJN_bishe/暴雨数据/00_DEM_Rain/B/gsmap_mvk/'
# # b_path = 'E:/LinJN_bishe/暴雨数据/00_DEM_Rain/B/station/'
# c_path = 'E:/LinJN_bishe/暴雨数据/00_DEM_Rain/B/standard/'
# d_path = 'E:/LinJN_bishe/暴雨数据/00_DEM_Rain/B/gsmap_gauge/'

# a_path = 'E:/LinJN_bishe/暴雨数据/00_DEM_Rain/C/gsmap_mvk/'
# # b_path = 'E:/LinJN_bishe/暴雨数据/00_DEM_Rain/C/station/'
# c_path = 'E:/LinJN_bishe/暴雨数据/00_DEM_Rain/C/standard/'
# d_path = 'E:/LinJN_bishe/暴雨数据/00_DEM_Rain/C/gsmap_gauge/'

#==============================================================================

train = True


a_names = read_datas(a_path)[0]
# b_names = read_datas(b_path)[0]
c_names = read_datas(c_path)[0]
d_names = read_datas(d_path)[0]

a_in = []
# b_in = []
c_in = []
d_in = []

path_random_save = 'F:/水文资料/各种数据/new_result/without_station/短时暴雨序列/00_DEM_Rain/'
mkdir(path_random_save)
# ==============================================================================
# 产生随机数
nums = np.arange(len(a_names))
np.random.shuffle(nums)

#保存随机数
with open(path_random_save + 'random2.txt','w')as f:
    for num in nums:
        f.writelines(str(num))
        f.writelines('\n')
#==============================================================================

#==============================================================================
# 读取随机数
nums = []
with open(path_random_save + 'random2.txt','r')as f:
    for line in f.readlines():
        nums.append(line.strip('\n'))
#==============================================================================

def save_datas(file_dir,file_name,file_in):
    for i in range(len(file_name)):
        data = []
        with open(file_dir + file_name[int(nums[i])],'r')as f:
            for line in f.readlines():
                # data.append(line.strip('\n').strip())
                lines = line.replace('\n', '').strip()
                if not len(lines):
                    continue
                else:
                    data.append(lines)
        file_in.append(data)
    return file_in


a_in = save_datas(a_path,a_names,a_in)
# b_in = save_datas(b_path,b_names,b_in)
c_in = save_datas(c_path,c_names,c_in)
d_in = save_datas(d_path,d_names,d_in)
# for i in range(len(a_names)):
#     data = []
#     with open(a_path + a_names[nums[i]],'r')as f:
#         for line in f.readlines():
#             data.append(line.strip('\n').strip())
#     a_in.append(data)
# a_in = np.array(a_in)
# a_in = a_in.astype('float64')
#
# #a_in = a_in.transpose()
#
# for i in range(len(a_names)):
#     data = []
#     with open(b_path + b_names[nums[i]],'r')as f:
#         for line in f.readlines():
#             data.append(line.strip('\n').strip())
#     b_in.append(data)
# b_in = np.array(b_in)
# b_in = b_in.astype('float64')
#
# #b_in = b_in.transpose()
#
# for i in range(len(a_names)):
#     data = []
#     with open(c_path + c_names[nums[i]],'r')as f:
#         for line in f.readlines():
#             data.append(line.strip('\n').strip())
#     c_in.append(data)
# c_in = np.array(c_in)
# c_in = c_in.astype('float64')

a_in = np.array(a_in)
a_in = a_in.astype('float64')


# b_in = np.array(b_in)
# b_in = b_in.astype('float64')

c_in = np.array(c_in)
c_in = c_in.astype('float64')

d_in = np.array(d_in)
d_in = d_in.astype('float64')

a = tf.placeholder(tf.float32, [None, 54])   # 每个txt含54小时的数据
# b = tf.placeholder(tf.float32, [None, 54])
c = tf.placeholder(tf.float32, [None, 54])
d = tf.placeholder(tf.float32, [None, 54])


ab = tf.concat([a],axis=-1)

#==============================================================================
# ab = tf.expand_dims(ab,axis=-1)
# rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=1)
# outputs,final_state = tf.nn.dynamic_rnn(
#     cell=rnn_cell,              # 选择传入的cell
#     inputs=ab,               # 传入的数据
#     initial_state=None,         # 初始状态
#     dtype=tf.float32,           # 数据类型
#     time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)，这里根据image结构选择False
# )
#==============================================================================
#print(outputs[:,1095:,:].shape)
# ab = tf.squeeze(outputs[:,1095:,:], -1)
# ab = tf.reshape(outputs[:,1095:,:], (-1,1))
#print(ab.shape)

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


c_ = add_layer(ab,54,54)
mse = tf.reduce_mean(tf.square(c_ - c))
train_op =tf.train.GradientDescentOptimizer(0.01).minimize(mse)

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)
saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=5)

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


num = int(len(a_names)*0.8)


#==============================================================================
# 训练时启用
path_train_save = path_random_save + 'modelnew\\'
mkdir(path_train_save)

while train:
    print('training!')
    #load(saver,sess,'E:/下载/lin/model365/model.ckpt-60000')

    for i in range(20001):
        loss,_ = sess.run([mse,train_op],feed_dict={a:a_in[:num,:],c:c_in[:num,:]})
        #print(i,loss)
        if i%1000 ==0:
            out = sess.run(c_,feed_dict={a:a_in[num:,:],c:c_in[num:,:]})
            out[out<0.1] = 0
            print("step:",i,"  RMSE:",math.sqrt(metrics.mean_squared_error(np.mean(out,axis=0), np.mean(c_in[num:,:],axis=0))))
            save(saver,sess,path_train_save,i)
    break
#==============================================================================
#==============================================================================
print('testing!')
load(saver,sess,path_train_save + 'model.ckpt-20000')
out = sess.run(c_,feed_dict={a:a_in[num:,:],c:c_in[num:,:]})
out[out<0.1]=0

path_predict_save =  path_random_save + 'predict\\'
mkdir(path_predict_save)

for i in range(len(a_names)-num):
    with open(path_predict_save+a_names[int(nums[num+i])].split('_')[1],'w')as f:
        for j in range(54):
            f.writelines(str(out[i,j]))
            f.writelines('\n')
#==============================================================================
# print(np.shape(np.mean(out,axis=1))) #读取矩阵长度
# print("RMSE:",math.sqrt(metrics.mean_squared_error(np.sum(out,axis=1), np.sum(c_in[num:,:],axis=1))))
# print("MAE:",metrics.mean_absolute_error(np.mean(out,axis=1), np.mean(c_in[num:,:],axis=1)))
# data2 = pd.DataFrame({'pre':np.mean(out,axis=1),
#                      'label':np.mean(c_in[num:,:],axis=1)})
# print('CC:',data2.corr())

number1 = out.shape[0]
number2 = out.shape[1]

RMSE_mvk = rmse_count(a_in[num:,:], c_in[num:,:], number1, number2)
print("RMSE_mvk:", RMSE_mvk)
ME_mvk = me_count(a_in[num:,:], c_in[num:,:], number1, number2)
print("ME_mvk:", ME_mvk)
CC_mvk = cc_count(a_in[num:,:], c_in[num:,:], number1, number2)
print("CC_mvk:", CC_mvk)
BIAS_mvk = bias_count(a_in[num:,:], c_in[num:,:], number1, number2)
print("BIAS_mvk:", BIAS_mvk)
print('*'*10)
RMSE_pre = rmse_count(out, c_in[num:,:], number1, number2)
print("RMSE_pre:", RMSE_pre)
ME_pre = me_count(out, c_in[num:,:], number1, number2)
print("ME_pre:", ME_pre)
CC_pre = cc_count(out, c_in[num:,:], number1, number2)
print("CC_pre:", CC_pre)
BIAS_pre = bias_count(out, c_in[num:,:], number1, number2)
print("BIAS_pre:", BIAS_pre)
print('*'*10)
RMSE_gauge = rmse_count(d_in[num:,:], c_in[num:,:], number1, number2)
print("RMSE_gauge:", RMSE_gauge)
ME_gauge = me_count(d_in[num:,:], c_in[num:,:], number1, number2)
print("ME_gauge:", ME_gauge)
CC_gauge = cc_count(d_in[num:,:], c_in[num:,:], number1, number2)
print("CC_pre:", CC_gauge)
BIAS_gauge = bias_count(d_in[num:,:], c_in[num:,:], number1, number2)
print("BIAS_pre:", BIAS_gauge)