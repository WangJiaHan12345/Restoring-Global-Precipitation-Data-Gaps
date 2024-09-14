#！/usr/bin/env python
# encoding: utf-8
'''
@author: 真梦行路
@file: tf_lstm2.py
@time: 2018/8/13 11:27
'''
###引入第三方模块###
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from numpy import *
from itertools import chain

###读取数据###
df=pd.read_csv(os.getcwd()+'\\data\\dataset_2.csv')
data=df.iloc[:,2:10].values
print(data)
###定义设置LSTM常量###
rnn_unit=20    #隐层单元的数量
input_size=6   #输入矩阵维度
output_size=2  #输出矩阵维度
lr=0.0006      #学习率
time_step=20   #设置时间步长
###制作带时间步长的训练集###
def get_train_data(batch_size=60,time_step=15,train_begin=0,train_end=5800):
    batch_index=[]
    data_train=data[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:6]
       y=normalized_train_data[i:i+time_step,6:]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y

###制作带时间步长的测试集###
def get_test_data(time_step=15,test_begin=5800):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample
    test_x,test_y=[],[]
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:6]
       y=normalized_test_data[i*time_step:(i+1)*time_step,6:]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,:6]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,6:]).tolist())
    return mean,std,test_x,test_y
#——————————————————定义LSTM网络权重和偏置——————————————————
#输入层、输出层权重、偏置

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,2]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[2,]))
       }

#——————————————————定义LSTM网络——————————————————
def lstm(X):
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell1=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    cell2=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    cell=tf.nn.rnn_cell.MultiRNNCell(cells=[cell1,cell2])
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    with tf.variable_scope('scope', reuse=tf.AUTO_REUSE):
        output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

#——————————————————LSTM模型训练——————————————————
def train_lstm(batch_size=60,time_step=15,train_begin=0,train_end=5800):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    print(train_y)
    pred,_=lstm(X)
    ###损失函数###
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1,2])-tf.reshape(Y, [-1,2])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=4)#只保留最后4次的模型参数
    # module_file = tf.train.latest_checkpoint(os.getcwd()+'\\data\\module')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, module_file)
        #重复训练10000次
        for i in range(1001):
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            if i % 100==0:
                print(i, loss_)
                print("保存模型：",saver.save(sess,os.getcwd()+'\\data\\module\\stock2.mode2',global_step=i))

#————————————————LSTM模型预测————————————————————
def prediction(time_step=15):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])#创建输入流图
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])#创建输出流图
    mean,std,test_x,test_y=get_test_data(time_step)
    pred,_=lstm(X)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        ###参数恢复，调用已经训练好的模型###
        module_file = tf.train.latest_checkpoint(os.getcwd()+'\\data\\module')
        saver.restore(sess, module_file)
        test_predict=[]
        for step in range(len(test_x)-1):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})
          predict=prob.reshape(-1,1)
          test_predict.extend(predict)

        ###循环输出其中一个预测变量###
        predict_test=[]
        for i in range(len(test_predict)):
            if i%2==0:
                predict_test.append(test_predict[i])
        ###循环输出测试原始数据###
        y_test=[]
        test_y=np.array(test_y).reshape(-1,1)
        for i in range(len(test_y)):
            if i%2==0:
                y_test.append(test_y[i])
        ###数据反归一化###
        test_y=np.array(y_test)*std[6]+mean[6]
        test_predict=np.array(predict_test)*std[6]+mean[6]
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差
        print(acc)
        #以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y,  color='r')
        plt.show()
train_lstm()
prediction()