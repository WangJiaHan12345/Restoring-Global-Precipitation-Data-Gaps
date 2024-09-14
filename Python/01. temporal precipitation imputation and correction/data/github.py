# ！/usr/bin/env python
# encoding: utf-8
###引入第三方模块###
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pandas as pd
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from numpy import *
from itertools import chain
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#

# 指标计算
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


def rmse_count(x_in, s_in,number1,number2):
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
            shang += x_in[i][j] - s_in[i][j]
    bias = (shang / xia) * 100
    return bias


def abias_count(x_in, s_in, number1, number2):
    # 求绝对偏差
    shang = 0
    xia = np.sum(x_in)
    for i in range(number1):
        for j in range(number2):
            shang += abs(x_in[i][j] - s_in[i][j])
    abias = (shang / xia) * 100
    return abias


def pod(x_in, s_in, number1, number2):
    # 求pod
    n10 = 0
    n01 = 0
    n11 = 0
    for i in range(number1):
        for j in range(number2):
        # CPC
            if s_in[i][j] >= 2.4 and x_in[i][j] < 2.4:
                n01 = n01 + 1
            # 卫星
            if x_in[i][j] >= 2.4 and s_in[i][j]< 2.4:
                n10 = n10 + 1
            #     cpc+卫星
            if x_in[i][j]>= 2.4 and s_in[i][j]>= 2.4:
                n11 = n11 +1

    pod = n11/(n11+n01)
    return pod

def far(x_in, s_in, number1, number2):
    # 求pod
    n10 = 0
    n01 = 0
    n11 = 0
    for i in range(number1):
        for j in range(number2):
            # CPC
            if s_in[i][j] >= 2.4 and x_in[i][j] < 2.4:
                n01 = n01 + 1
            # 卫星
            if x_in[i][j] >= 2.4 and s_in[i][j] < 2.4:
                n10 = n10 + 1
            #     cpc+卫星
            if x_in[i][j] >= 2.4 and s_in[i][j] >= 2.4:
                n11 = n11 + 1
    far = n10/(n11+n10)
    return far

def csi(x_in, s_in, number1, number2):
    # 求pod
    n10 = 0
    n01 = 0
    n11 = 0
    for i in range(number1):
        for j in range(number2):
            # CPC
            if s_in[i][j] >= 2.4 and x_in[i][j] < 2.4:
                n01 = n01 + 1
            # 卫星
            if x_in[i][j] >= 2.4 and s_in[i][j] < 2.4:
                n10 = n10 + 1
            #     cpc+卫星
            if x_in[i][j] >= 2.4 and s_in[i][j] >= 2.4:
                n11 = n11 + 1

    csi = n11/(n11+n10+n01)
    return csi


grids= 416  #网格数
###读取数据###
# 训练数据
a_path = 'H:\\时间预测\\结果\\shirun\\02_final_data\\全年\\xunlian_1\\Early\\'
c_path = 'H:\\时间预测\\结果\\shirun\\02_final_data\\全年\\xunlian_1\\cpc\\'
d_path = 'H:\\时间预测\\结果\\shirun\\02_final_data\\全年\\xunlian_1\\Final\\'


train = True

a_names = read_datas(a_path)[0]
c_names = read_datas(c_path)[0]
d_names = read_datas(d_path)[0]

a_in = []
c_in = []
d_in = []

a_in = save_datas(a_path, a_names, a_in)
c_in = save_datas(c_path, c_names, c_in)
d_in = save_datas(d_path, d_names, d_in)


# 还没保存结果
path_random_save = 'H:\\时间预测\\结果\\shirun\\result\\'
mkdir(path_random_save)  # 创建一个目录存放结果


a_in = np.array(a_in)
a_in = a_in.astype('float32')
# a_in = list(chain.from_iterable(a_in))
# a_in = np.array([a_in]).T
a_in = np.transpose(a_in)
# print(a_in)
# print(a_in.shape[0])
# print(a_in.shape[1])


c_in = np.array(c_in)
c_in = c_in.astype('float32')
# c_in = list(chain.from_iterable(c_in))
# c_in = np.array([c_in]).T
c_in = np.transpose(c_in)


d_in = np.array(d_in)
d_in = d_in.astype('float32')
# d_in = list(chain.from_iterable(d_in))
# d_in = np.array([d_in]).T
d_in = np.transpose(d_in)



count = grids + grids
train_data = np.zeros((1096, count))  #276x9542   11170x2
train_data[:, :grids] = a_in[:, :]
train_data[:, grids:] = c_in[:, :]

# 测试数据
a_path_1 = 'H:\\时间预测\\结果\\shirun\\02_final_data\\全年\\ceshi_1\\Early\\'
c_path_1 = 'H:\\时间预测\\结果\\shirun\\02_final_data\\全年\\ceshi_1\\cpc\\'
d_path_1 = 'H:\\时间预测\\结果\\shirun\\02_final_data\\全年\\ceshi_1\\Final\\'

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
# a_in_1 = list(chain.from_iterable(a_in_1))
# a_in_1 = np.array([a_in_1]).T
a_in_1 = np.transpose(a_in_1)
a_in_1_T = np.transpose(a_in_1)
print(a_in_1_T.shape[0])
print(a_in_1_T.shape[1])



c_in_1 = np.array(c_in_1)
c_in_1 = c_in_1.astype('float32')
# c_in_1 = list(chain.from_iterable(c_in_1))
# c_in_1 = np.array([c_in_1]).T
c_in_1 = np.transpose(c_in_1)
c_in_1_T = np.transpose(c_in_1)

d_in_1 = np.array(d_in_1)
d_in_1 = d_in_1.astype('float32')
# d_in_1 = list(chain.from_iterable(d_in_1))
# d_in_1 = np.array([d_in_1]).T
d_in_1 = np.transpose(d_in_1)
d_in_1_T = np.transpose(d_in_1)

test_data = np.zeros((365, count))  # 9542x92
test_data[:, :grids] = a_in_1[:, :]
test_data[:, grids:] = c_in_1[:, :]

###定义设置LSTM常量###
rnn_unit = 500  # 隐层单元的数量
input_size = grids  # 输入矩阵维度
output_size = grids  # 输出矩阵维度
lr = 0.001  # 学习率


###制作带时间步长的训练集###
#如果你的数据有10000行，训练100次把所有数据训练完，那么你的batch_size=10000/100=100  所以这里应该是天数
#之前的batch_size中只是规定了一个每次feed多少行数据进去，并没有涵盖一个时间的概念进去，
# 而这个参数刚好就是对于时间的限制，毕竟你是做时间序列预测，所以才多了这个参数。
# 换句话说，就是在一个batch_size中，你要定义一下每次数据的时间序列是多少？
# 如果你的数据都是按照时间排列的，batch_size是100的话，time_step=10
# 在第1次训练的时候，是用前100行数据进行训练，而在这其中每次给模型10个连续时间序列的数据。
# 那你是不是以为应该是1-10，11-20，21-30，这样把数据给模型？还是不对，请看下图。
# time_step=n, 就意味着我们认为每一个值都和它前n个值有关系
def get_train_data(batch_size=365, time_step=5): #90是按照格网数来的
    batch_index = []
    train_x, train_y = [], []  # 训练集
    for i in range(len(train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = train_data[i:i + time_step, :grids]
        y = train_data[i:i + time_step, grids:]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(train_data) - time_step))
    return batch_index, train_x, train_y


###制作带时间步长的测试集###
def get_test_data(time_step=5):
    mean = np.mean(test_data, axis=0)
    std = np.std(test_data, axis=0)
    size = (len(test_data) + time_step - 1) // time_step  # 有size个sample
    test_x, test_y = [], []
    for i in range(size - 1):
        x = test_data[i * time_step:(i + 1) * time_step, :grids]
        y = test_data[i * time_step:(i + 1) * time_step, grids:]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((test_data[(i + 1) * time_step:, :grids]).tolist())
    test_y.extend((test_data[(i + 1) * time_step:, grids:]).tolist())
    return mean, std, test_x, test_y


# ——————————————————定义LSTM网络权重和偏置——————————————————
# 输入层、输出层权重、偏置

weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, grids]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[grids, ]))
}

# ——————————————————定义LSTM网络——————————————————
def lstm(X):
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states


# ——————————————————LSTM模型训练—————————————————
#——————————————————训练模型——————————————————
path_train_save = path_random_save + 'modelnew\\'
mkdir(path_train_save)

def train_lstm(batch_size=365,time_step=5):

    X = tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y = tf.placeholder(tf.float32, shape=[None,time_step,output_size])

    batch_index,train_x,train_y=get_train_data(batch_size,time_step)

    pred,_=lstm(X)
    #损失函数
    # loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1,grids])-tf.reshape(Y, [-1,grids])))
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))

    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=1)
    # module_file = tf.train.latest_checkpoint()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, module_file)
        #重复训练2000次
        for i in range(3001):
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            if i % 500==0:
                print(i, loss_)
                print("保存模型：",saver.save(sess,path_train_save+'model.ckpt',global_step=i))



# ————————————————LSTM模型预测————————————————————

def test_lstm(time_step=5):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])  # 创建输入流图
    mean, std, test_x, test_y = get_test_data(time_step)
    pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        ###参数恢复，调用已经训练好的模型###
        module_file = tf.train.latest_checkpoint(path_train_save)
        saver.restore(sess, module_file)
        # load(saver, sess, path_train_save + 'model.ckpt-500')

        test_predict = []

        test_predict = sess.run(pred, feed_dict={X: [test_x[0]]})
        test_predict = np.transpose(test_predict)
        test_predict[test_predict < 0] = 0

        for step in range(1,len(test_x)):
            prob = sess.run(pred, feed_dict={X: [test_x[step]]})
            prob = np.transpose(prob)
            prob[prob < 0] = 0
            # test_predict =  prob
            # # predict = prob.reshape(-1, 1)
            # test_predict.extend(predict)
            test_predict = np.hstack((test_predict, prob))  # 水平组合




        number1 = test_predict.shape[0]
        number2 = test_predict.shape[1]
        print(number1,number2)



        CC_early = cc_count(a_in_1_T, c_in_1_T, number1, number2)
        print("CC_early :", CC_early)
        BIAS_early = bias_count(a_in_1_T, c_in_1_T, number1, number2)
        print("BIAS_early :", BIAS_early)
        RMSE_early = rmse_count(a_in_1_T, c_in_1_T, number1, number2)
        print("RMSE_early :", RMSE_early)
        ABIAS_early = abias_count(a_in_1_T, c_in_1_T, number1, number2)
        print("ABIAS_early :", ABIAS_early)
        POD_early = pod(a_in_1_T, c_in_1_T, number1, number2)
        print("POD_early :", POD_early)
        FAR_early = far(a_in_1_T, c_in_1_T, number1, number2)
        print("FAR_early :", FAR_early)
        CSI_early = csi(a_in_1_T, c_in_1_T, number1, number2)
        print("CSI_early :", CSI_early)
        print('*' * 10)

        CC_pre = cc_count(test_predict, c_in_1_T, number1, number2)
        print("CC_pre:", CC_pre)
        BIAS_pre = bias_count(test_predict, c_in_1_T, number1, number2)
        print("BIAS_pre:", BIAS_pre)
        RMSE_pre = rmse_count(test_predict, c_in_1_T, number1, number2)
        print("RMSE_pre:", RMSE_pre)
        ABIAS_pre = abias_count(test_predict, c_in_1_T, number1, number2)
        print("ABIAS_pre:", ABIAS_pre)
        POD_pre = pod(test_predict, c_in_1_T, number1, number2)
        print("POD_pre:", POD_pre)
        FAR_pre = far(test_predict, c_in_1_T,number1, number2)
        print("FAR_pre:", FAR_pre)
        CSI_pre = csi(test_predict, c_in_1_T, number1, number2)
        print("CSI_pre:", CSI_pre)
        print('*' * 10)

        CC_final = cc_count(d_in_1_T,c_in_1_T, number1, number2)
        print("CC_final:", CC_final)
        BIAS_final = bias_count(d_in_1_T,c_in_1_T, number1, number2)
        print("BIAS_final:", BIAS_final)
        RMSE_final = rmse_count(d_in_1_T,c_in_1_T, number1, number2)
        print("RMSE_final:", RMSE_final)
        ABIAS_final = abias_count(d_in_1_T,c_in_1_T, number1, number2)
        print("ABIAS_final:", ABIAS_final)
        POD_final = pod(d_in_1_T,c_in_1_T, number1, number2)
        print("POD_final:", POD_final)
        FAR_final = far(d_in_1_T,c_in_1_T, number1, number2)
        print("FAR_final:", FAR_final)
        CSI_final = csi(d_in_1_T,c_in_1_T, number1, number2)
        print("CSI_final:", CSI_final, number1, number2)


train_lstm()
print("开始预测")
tf.compat.v1.get_variable_scope().reuse_variables()
test_lstm()