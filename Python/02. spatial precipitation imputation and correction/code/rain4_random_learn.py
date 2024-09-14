# -*- coding: utf-8 -*-
"""
@author: ZZY
"""
import random
import math
# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
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




months = ['6-8']
for month in months:
    RMSE_early_sum, ME_early_sum, CC_early_sum, BIAS_early_sum = 0, 0, 0, 0
    RMSE_pre_sum, ME_pre_sum, CC_pre_sum, BIAS_pre_sum = 0, 0, 0, 0
    RMSE_final_sum, ME_final_sum, CC_final_sum, BIAS_final_sum = 0, 0, 0, 0
    for k in range(10):
        if __name__ == '__main__':
            a_path = 'H:\\banshirun\\02_final_data\\Early4\\' + month + '\\'
            # b_path = 'E:/LinJN_bishe/暴雨数据/00_DEM_Rain/B/station/'
            c_path = 'H:\\banshirun\\02_final_data\\cpc4\\' + month + '\\'
            d_path = 'H:\\banshirun\\02_final_data\\Final4\\' + month + '\\'

            train = True

            a_names = read_datas(a_path)[0]  ##这里得到的是各个mvk_xx的名字
            # b_names = read_datas(b_path)[0]
            c_names = read_datas(c_path)[0]
            d_names = read_datas(d_path)[0]
            # print(a_names)

            a_in = []  # 用来存放各个mvk里的数据
            # b_in = []
            c_in = []
            d_in = []

            n = str(k + 1)
            # path_random_save = 'H:\\banganhan\\Early_n\\result_sumrain_12-2\\' + num + '\\'+ n + '\\'
            path_random_save = 'H:\\banshirun\\global1\\' + month + '\\' + n + '\\'
            mkdir(path_random_save)  # 创建一个目录存放结果
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
            # print(a_in)
            # b_in = save_datas(b_path,b_names,b_in,nums)
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

            if  month == '6-8' or month == '3-5':
                count = 368
            elif  month == '9-11':
                count = 364
            else:
                count = 361

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
            saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=2)

            num = int(len(a_names) * 0.8)  # 可换0.8  0.7
            # num1 = int(len(a_names) * 0.8)

            # ==============================================================================
            # 训练时启用
            path_train_save = path_random_save + 'modelnew\\'
            mkdir(path_train_save)
            string_name = ''
            while train:
                print('training!')

                for i in range(9001):
                    sess.run(train_op, feed_dict={a: a_in[:num, :], c: c_in[:num, :], keep_prob: 0.8})
                    # sess.run(train_op, feed_dict={d: d_in[:num, :], c: c_in[:num, :], keep_prob: 0.8})
                    # print(i,loss)
                    if i % 1000 == 0:
                        loss_train, out1 = sess.run([mse, predict],
                                                    feed_dict={a: a_in[:num, :], c: c_in[:num, :], keep_prob: 1})
                        # loss_train, out1 = sess.run([mse, predict], feed_dict={d: d_in[:num, :], c: c_in[:num, :], keep_prob: 1})
                        out1[out1 < 0] = 0
                        loss_test, out2 = sess.run([mse, predict],
                                                   feed_dict={a: a_in[num:, :], c: c_in[num:, :], keep_prob: 1})
                        # loss_test, out2 = sess.run([mse, predict],feed_dict={d: d_in[num:, :], c: c_in[num:, :], keep_prob: 1})
                        out2[out2 < 0] = 0
                        cc_test = cc_count(out2, c_in[num:, :], out2.shape[0], out2.shape[1])
                        cc_train = cc_count(out1, c_in[:num, :], out1.shape[0], out1.shape[1])
                        print("step", i, "  loss_train:", loss_train, "  loss_test:", loss_test, "CC1", cc_train, "CC2",
                              cc_test)
                        save(saver, sess, path_train_save, i)
                        string_name = 'model.ckpt-' + str(i)
                break

            print('testing')
            load(saver, sess, path_train_save + str(string_name))
            out = sess.run(predict, feed_dict={a: a_in[num:, :], c: c_in[num:, :], keep_prob: 1})
            # out = sess.run(predict, feed_dict={d: d_in[num:, :], c: c_in[num:, :], keep_prob: 1})
            out[out < 0] = 0

            path_predict_save = path_random_save + 'predict\\'
            mkdir(path_predict_save)

            for i in range(len(a_names) - num):
                with open(path_predict_save + str(a_names[int(nums[num + i])].split('_')[1]), 'w')as f:
                    for j in range(count):
                        f.writelines(str(out[i, j]))
                        f.writelines('\n')

            number1 = out.shape[0]
            number2 = out.shape[1]

            CC_early = cc_count(a_in[num:, :], c_in[num:, :], number1, number2)
            print("CC_early :", CC_early)
            CC_early_sum = CC_early_sum + CC_early
            BIAS_early = bias_count(a_in[num:, :], c_in[num:, :], number1, number2)
            print("BIAS_early :", BIAS_early)
            BIAS_early_sum = BIAS_early_sum + BIAS_early
            RMSE_early = rmse_count(a_in[num:, :], c_in[num:, :], number1, number2)
            print("RMSE_early :", RMSE_early)
            RMSE_early_sum = RMSE_early_sum + RMSE_early
            ME_early = me_count(a_in[num:, :], c_in[num:, :], number1, number2)
            print("ME_early :", ME_early)
            ME_early_sum = ME_early_sum + ME_early
            print('*' * 10)
            CC_pre = cc_count(out, c_in[num:, :], number1, number2)
            print("CC_pre:", CC_pre)
            CC_pre_sum = CC_pre_sum + CC_pre
            BIAS_pre = bias_count(out, c_in[num:, :], number1, number2)
            print("BIAS_pre:", BIAS_pre)
            BIAS_pre_sum = BIAS_pre_sum + BIAS_pre
            RMSE_pre = rmse_count(out, c_in[num:, :], number1, number2)
            print("RMSE_pre:", RMSE_pre)
            RMSE_pre_sum = RMSE_pre_sum + RMSE_pre
            ME_pre = me_count(out, c_in[num:, :], number1, number2)
            print("ME_pre:", ME_pre)
            ME_pre_sum = ME_pre_sum + ME_pre
            print('*' * 10)
            CC_final = cc_count(d_in[num:, :], c_in[num:, :], number1, number2)
            print("CC_final:", CC_final)
            CC_final_sum = CC_final_sum + CC_final
            BIAS_final = bias_count(d_in[num:, :], c_in[num:, :], number1, number2)
            print("BIAS_final:", BIAS_final)
            BIAS_final_sum = BIAS_final_sum + BIAS_final
            RMSE_final = rmse_count(d_in[num:, :], c_in[num:, :], number1, number2)
            print("RMSE_final:", RMSE_final)
            RMSE_final_sum = RMSE_final_sum + RMSE_final
            ME_final = me_count(d_in[num:, :], c_in[num:, :], number1, number2)
            print("ME_final:", ME_final)
            ME_final_sum = ME_final_sum + ME_final

    CC_early = CC_early_sum / 10
    BIAS_early = BIAS_early_sum / 10
    RMSE_early = RMSE_early_sum / 10
    ME_early = ME_early_sum / 10
    CC_pre = CC_pre_sum / 10
    BIAS_pre = BIAS_pre_sum / 10
    RMSE_pre = RMSE_pre_sum / 10
    ME_pre = ME_pre_sum / 10
    CC_final = CC_final_sum / 10
    BIAS_final = BIAS_final_sum / 10
    RMSE_final = RMSE_final_sum / 10
    ME_final = ME_final_sum / 10

    print('#' * 10)
    print("CC_early :", CC_early)
    print("BIAS_early :", BIAS_early)
    print("RMSE_early :", RMSE_early)
    print("ME_early :", ME_early)
    print('*' * 10)
    print("CC_pre:", CC_pre)
    print("BIAS_pre:", BIAS_pre)
    print("RMSE_pre:", RMSE_pre)
    print("ME_pre:", ME_pre)
    print('*' * 10)
    print("CC_final:", CC_final)
    print("BIAS_final:", BIAS_final)
    print("RMSE_final:", RMSE_final)
    print("ME_final:", ME_final)
# #     #
# #
# ###########################################################################################
