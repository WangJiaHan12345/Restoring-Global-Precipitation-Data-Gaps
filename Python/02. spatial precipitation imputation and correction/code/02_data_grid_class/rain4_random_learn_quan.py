# -*- coding: utf-8 -*-
"""
@author: ZZY
"""
import random
import math
# import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
# tf.enable_eager_execution()
tf.disable_v2_behavior()
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import os
# import cv2
import numpy as np
# import tensorflow.contrib.slim as slim
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


def save_datas(file_dir, file_name, file_in, nums):
    for i in range(len(file_name)):
        data = []
        with open(file_dir + file_name[int(nums[i])], 'r')as f:
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


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


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


def mae_count(x_in, s_in, number1, number2):
    sum = 0
    count = 0
    for i in range(number1):
        for j in range(number2):
            count += 1
            cha = abs(x_in[i][j] - s_in[i][j])
            sum += cha
    mae = sum / count
    return mae


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
    xia = np.sum(s_in)
    for i in range(number1):
        for j in range(number2):
            shang += x_in[i][j]-s_in[i][j]
    bias = (shang / xia) * 100
    return bias

def abias_count(x_in, s_in, number1, number2):
    # 求绝对偏差
    shang = 0
    xia = np.sum(s_in)
    for j in range(number1):
        for k in range(number2):
            shang += abs(x_in[j][k]-s_in[j][k])
    abias = (shang / xia) * 100
    return abias


def pod(x_in, s_in, number1, number2):
    # 求pod
    n10 = 0
    n01 = 0
    n11 = 0
    for j in range(number1):
        for k in range(number2):
        # CPC
            if s_in[j][k] >= 2.4 and x_in[j][k] < 2.4:
                n01 = n01 + 1
            # 卫星
            if x_in[j][k] >= 2.4 and s_in[j][k] < 2.4:
                n10 = n10 + 1
            #     cpc+卫星
            if x_in[j][k] >= 2.4 and s_in[j][k] >= 2.4:
                n11 = n11 +1

    pod = n11/(n11+n01)
    return pod

def far(x_in, s_in, number1, number2):
    # 求pod
    n10 = 0
    n01 = 0
    n11 = 0
    for j in range(number1):
        for k in range(number2):
            # CPC
            if s_in[j][k] >= 2.4 and x_in[j][k] < 2.4:
                n01 = n01 + 1
            # 卫星
            if x_in[j][k] >= 2.4 and s_in[j][k] < 2.4:
                n10 = n10 + 1
            #     cpc+卫星
            if x_in[j][k] >= 2.4 and s_in[j][k] >= 2.4:
                n11 = n11 + 1
    far = n10/(n11+n10)
    return far

def csi(x_in, s_in, number1, number2):
    # 求pod
    n10 = 0
    n01 = 0
    n11 = 0
    for j in range(number1):
        for k in range(number2):
            # CPC
            if s_in[j][k] >= 2.4 and x_in[j][k] < 2.4:
                n01 = n01 + 1
            # 卫星
            if x_in[j][k] >= 2.4 and s_in[j][k] < 2.4:
                n10 = n10 + 1
            #     cpc+卫星
            if x_in[j][k] >= 2.4 and s_in[j][k] >= 2.4:
                n11 = n11 + 1

    csi = n11/(n11+n10+n01)
    return csi

seasons = ['3-5','6-8','9-11','12-2']
for season in seasons:
    blocks = ['F']
    for block in blocks:
        RMSE_pre_sum, MAE_pre_sum, CC_pre_sum, BIAS_pre_sum, ABIAS_pre_sum, POD_pre_sum, FAR_pre_sum, CSI_pre_sum = 0, 0, 0, 0, 0, 0, 0, 0
        RMSE_mvk_sum, MAE_mvk_sum, CC_mvk_sum, BIAS_mvk_sum, ABIAS_mvk_sum, POD_mvk_sum, FAR_mvk_sum, CSI_mvk_sum = 0, 0, 0, 0, 0, 0, 0, 0
        RMSE_gauge_sum, MAE_gauge_sum, CC_gauge_sum, BIAS_gauge_sum, ABIAS_gauge_sum, POD_gauge_sum, FAR_gauge_sum, CSI_gauge_sum = 0, 0, 0, 0, 0, 0, 0, 0
        l = 10
        for k in range(l):
            if __name__ == '__main__':
                a_path = 'H:\\青藏高原数据\\空间+时间\\01_data\\2015-2016.5\\gsmap_mvk\\'+ season + '\\'
                # b_path = 'E:/LinJN_bishe/暴雨数据/00_DEM_Rain/B/station/'
                c_path = 'H:\\青藏高原数据\\空间+时间\\01_data\\2015-2016.5\\国家气象局\\'+ season + '\\'
                d_path = 'H:\\青藏高原数据\\空间+时间\\01_data\\2015-2016.5\\gsmap_mvk\\'+ season + '\\'

                train = True

                a_names = read_datas(a_path)[0]  ##这里得到的是各个mvk_xx的名字
                # b_names = read_datas(b_path)[0]
                c_names = read_datas(c_path)[0]
                d_names = read_datas(d_path)[0]
                # print(a_names)

                a_in = [] #用来存放各个mvk里的数据
                # b_in = []
                c_in = []
                d_in = []

                n = str(k+1)
                path_random_save = 'H:\\青藏高原数据\\空间+时间\\01_data\\2015-2016.5\\result\\' + season + '\\' + str(k) + '\\'
                mkdir(path_random_save) #创建一个目录存放结果
                # ==============================================================================
                # 产生随机数
                nums = np.arange(len(a_names))
                np.random.shuffle(nums)

                # 保存随机数
                with open(path_random_save + 'random.txt', 'w')as f:
                    for num in nums:
                        f.writelines(str(num))
                        f.writelines('\n')
                # ==============================================================================

                # ==============================================================================
                # 读取随机数
                nums = []
                with open(path_random_save + 'random.txt', 'r')as f:
                    for line in f.readlines():
                        nums.append(line.strip('\n'))
                # ==============================================================================

                a_in = save_datas(a_path, a_names, a_in, nums)
                c_in = save_datas(c_path, c_names, c_in, nums)
                d_in = save_datas(d_path, d_names, d_in, nums)


                a_in = np.array(a_in)
                # print(a_in)
                a_in = a_in.astype('float64')


                # b_in = np.array(b_in)
                # b_in = b_in.astype('float64')

                c_in = np.array(c_in)
                c_in = c_in.astype('float64')
                # print(c_in)

                d_in = np.array(d_in)
                d_in = d_in.astype('float64')

                if season == '3-5':
                    count = 153
                elif season == '6-8':
                    count = 153
                elif season == '9-11':
                    count = 152
                else:
                    count = 150



                keep_prob = tf.placeholder(tf.float32)
                a = tf.placeholder(tf.float32, [None, count])
                c = tf.placeholder(tf.float32, [None, count])
                d = tf.placeholder(tf.float32, [None, count])


                l1 = add_layer(a, count, count, activation_function=tf.nn.tanh)  # 激励函数
                # l1 = add_layer(d, 368, 368, activation_function=tf.nn.tanh)  # 激励函数
                # l2 = add_layer(l1,40 ,40,activation_function=tf.nn.tanh)
                predict = add_layer(l1, count, count, activation_function=None)
                # predict = add_layer(a, 361, 361, activation_function=tf.nn.sigmoid)

                lr = tf.Variable(0.001, dtype=tf.float32)
                mse = tf.reduce_mean(tf.square(predict - c))
                # train_op = tf.train.GradientDescentOptimizer(0.01).minimize(mse)
                train_op = tf.train.AdamOptimizer(lr).minimize(mse)

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                sess = tf.Session(config=config)
                init = tf.global_variables_initializer()

                sess.run(init)
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)

                num = int(len(a_names) * 0.8)   #可换0.8  0.7

                # ==============================================================================
                # 训练时启用
                path_train_save = path_random_save + 'modelnew\\'
                mkdir(path_train_save)
                string_name = ''
                while train:
                    print('training!')

                    for i in range(15001):
                        loss,_ = sess.run([mse,train_op], feed_dict={a: a_in[:num, :], c: c_in[:num, :], keep_prob:0.8})
                        if i % 5000 == 0:
                            loss_train,out1 = sess.run([mse,predict], feed_dict={a:a_in[:num,:], c:c_in[:num,:], keep_prob:1})
                            out1[out1 < 0] = 0
                            loss_test,out2 = sess.run([mse,predict], feed_dict={a: a_in[num:, :], c: c_in[num:, :], keep_prob:1})
                            out2[out2 < 0] = 0
                            cc_test = cc_count(out2, c_in[num:, :], out2.shape[0],out2.shape[1])
                            cc_train = cc_count(out1, c_in[:num, :], out1.shape[0], out1.shape[1])
                            print("step", i, "  loss_train:", loss_train,"  loss_test:", loss_test,"CC1",cc_train,"CC2",cc_test)
                            save(saver, sess, path_train_save, i)
                            string_name = 'model.ckpt-' + str(i)
                    break

                print('testing')
                load(saver, sess, path_train_save + str(string_name))
                out = sess.run(predict, feed_dict={a: a_in[num:, :], c: c_in[num:, :], keep_prob:1})
                out[out < 0] = 0

                path_predict_save = path_random_save + 'predict\\'
                mkdir(path_predict_save)

                for i in range(len(a_names)-num):
                    with open(path_predict_save + str(a_names[int(nums[num+i])].split('_')[0]), 'w')as f:
                        for j in range(count):
                            f.writelines(str(out[i][j]))
                            f.writelines('\n')


                number1 = out.shape[0]
                number2 = out.shape[1]

                CC_mvk = cc_count(a_in[num:, :], c_in[num:, :], number1, number2)
                CC_mvk_sum = CC_mvk_sum + CC_mvk
                BIAS_mvk = bias_count(a_in[num:, :], c_in[num:, :], number1, number2)
                BIAS_mvk_sum = BIAS_mvk_sum + BIAS_mvk
                ABIAS_mvk = abias_count(a_in[num:, :], c_in[num:, :], number1, number2)
                ABIAS_mvk_sum = ABIAS_mvk_sum + ABIAS_mvk
                RMSE_mvk = rmse_count(a_in[num:, :], c_in[num:, :], number1, number2)
                RMSE_mvk_sum = RMSE_mvk_sum + RMSE_mvk
                MAE_mvk = mae_count(a_in[num:, :], c_in[num:, :], number1, number2)
                MAE_mvk_sum = MAE_mvk_sum + MAE_mvk
                POD_mvk = pod(a_in[num:, :], c_in[num:, :], number1, number2)
                POD_mvk_sum = POD_mvk_sum + POD_mvk
                FAR_mvk = far(a_in[num:, :], c_in[num:, :], number1, number2)
                FAR_mvk_sum = FAR_mvk_sum + FAR_mvk
                CSI_mvk = csi(a_in[num:, :], c_in[num:, :], number1, number2)
                CSI_mvk_sum = CSI_mvk_sum + CSI_mvk

                CC_pre = cc_count(out, c_in[num:, :], number1, number2)
                CC_pre_sum = CC_pre_sum + CC_pre
                BIAS_pre = bias_count(out, c_in[num:, :], number1, number2)
                BIAS_pre_sum = BIAS_pre_sum + BIAS_pre
                ABIAS_pre = abias_count(out, c_in[num:, :], number1, number2)
                ABIAS_pre_sum = ABIAS_pre_sum + ABIAS_pre
                RMSE_pre = rmse_count(out, c_in[num:, :], number1, number2)
                RMSE_pre_sum = RMSE_pre_sum + RMSE_pre
                MAE_pre = mae_count(out, c_in[num:, :], number1, number2)
                MAE_pre_sum = MAE_pre_sum + MAE_pre
                POD_pre = pod(out, c_in[num:, :], number1, number2)
                POD_pre_sum = POD_pre_sum + POD_pre
                FAR_pre = far(out, c_in[num:, :], number1, number2)
                FAR_pre_sum = FAR_pre_sum + FAR_pre
                CSI_pre = csi(out, c_in[num:, :], number1, number2)
                CSI_pre_sum = CSI_pre_sum + CSI_pre

                CC_gauge = cc_count(d_in[num:, :], c_in[num:, :], number1, number2)
                CC_gauge_sum = CC_gauge_sum + CC_gauge
                BIAS_gauge = bias_count(d_in[num:, :], c_in[num:, :], number1, number2)
                BIAS_gauge_sum = BIAS_gauge_sum + BIAS_gauge
                ABIAS_gauge = abias_count(d_in[num:, :], c_in[num:, :], number1, number2)
                ABIAS_gauge_sum = ABIAS_gauge_sum + ABIAS_gauge
                RMSE_gauge = rmse_count(d_in[num:, :], c_in[num:, :], number1, number2)
                RMSE_gauge_sum = RMSE_gauge_sum + RMSE_gauge
                MAE_gauge = mae_count(d_in[num:, :], c_in[num:, :], number1, number2)
                MAE_gauge_sum = MAE_gauge_sum + MAE_gauge
                POD_gauge = pod(d_in[num:, :], c_in[num:, :], number1, number2)
                POD_gauge_sum = POD_gauge_sum + POD_gauge
                FAR_gauge = far(d_in[num:, :], c_in[num:, :], number1, number2)
                FAR_gauge_sum = FAR_gauge_sum + FAR_gauge
                CSI_gauge = csi(d_in[num:, :], c_in[num:, :], number1, number2)
                CSI_gauge_sum = CSI_gauge_sum + CSI_gauge

        CC_pre = CC_pre_sum / l
        BIAS_pre = BIAS_pre_sum / l
        ABIAS_pre = ABIAS_pre_sum / l
        RMSE_pre = RMSE_pre_sum / l
        MAE_pre = MAE_pre_sum / l
        POD_pre = POD_pre_sum / l
        FAR_pre = FAR_pre_sum / l
        CSI_pre = CSI_pre_sum / l

        CC_mvk = CC_mvk_sum / l
        BIAS_mvk = BIAS_mvk_sum / l
        ABIAS_mvk = ABIAS_mvk_sum / l
        RMSE_mvk = RMSE_mvk_sum / l
        MAE_mvk = MAE_mvk_sum / l
        POD_mvk = POD_mvk_sum / l
        FAR_mvk = FAR_mvk_sum / l
        CSI_mvk = CSI_mvk_sum / l

        CC_gauge = CC_gauge_sum / l
        BIAS_gauge = BIAS_gauge_sum / l
        ABIAS_gauge = ABIAS_gauge_sum / l
        RMSE_gauge = RMSE_gauge_sum / l
        MAE_gauge = MAE_gauge_sum / l
        POD_gauge = POD_gauge_sum / l
        FAR_gauge = FAR_gauge_sum / l
        CSI_gauge = CSI_gauge_sum / l

        print(season)
        print(block)
        print('#' * 10)
        print("CC_mvk :", CC_mvk)
        print("BIAS_mvk :", BIAS_mvk)
        print("ABIAS_mvk :", ABIAS_mvk)
        print("RMSE_mvk :", RMSE_mvk)
        print("MAE_mvk :", MAE_mvk)
        print("POD_mvk :", POD_mvk)
        print("FAR_mvk :", FAR_mvk)
        print("CSI_mvk :", CSI_mvk)
        print('*' * 10)
        print("CC_pre :", CC_pre)
        print("BIAS_pre :", BIAS_pre)
        print("ABIAS_pre :", ABIAS_pre)
        print("RMSE_pre :", RMSE_pre)
        print("MAE_pre :", MAE_pre)
        print("POD_pre :", POD_pre)
        print("FAR_pre :", FAR_pre)
        print("CSI_pre :", CSI_pre)
        print('*' * 10)
        print("CC_gauge :", CC_gauge)
        print("BIAS_gauge :", BIAS_gauge)
        print("ABIAS_gauge :", ABIAS_gauge)
        print("RMSE_gauge :", RMSE_gauge)
        print("MAE_gauge :", MAE_gauge)
        print("POD_gauge :", POD_gauge)
        print("FAR_gauge :", FAR_gauge)
        print("CSI_gauge :", CSI_gauge)
        print('*' * 10)