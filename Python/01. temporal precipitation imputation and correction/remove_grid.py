import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import shutil


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


# 文件移除

# 改路径
a_path = 'G:\\全球\\空间校正结果\\用于和GPCP比较\\月尺度\\更小范围内的\\DNN_2.5\\' #改路径   国家气象局
b_path = 'G:\\全球\\空间校正结果\\用于和GPCP比较\\月尺度\\更小范围内的\\Final_2.5\\' #改路径
# c_path = 'H:\\青藏高原数据\空间+时间\\01_data\\2015-2016\\result\\1-12\\' #改路径
# d_path = 'F:\\青藏高原\\空间\\年度\\12-2\\ANN\\'
# train = True

a_names = read_datas(a_path)[0]
b_names = read_datas(b_path)[0]
# c_names = read_datas(c_path)[0]
# d_names = read_datas(d_path)[0]
# e_names = read_datas(e_path)[0]


# 在同一个文件夹下进行删除
for i in range(len(b_names)):
    if b_names[i] in a_names:
        continue
    else:
        remove_path = b_path + b_names[i]   #改路径
        os.remove(remove_path)

# for i in range(len(b_names)):
#     if b_names[i] in a_names:
#         continue
#     else:
#         remove_path = b_path + b_names[i]   #改路径
#         os.remove(remove_path)


# e_path = 'H:\\青藏高原数据\\时间预测\\2015-2017\\result(不分块)\\随机部分日期数据\\ANN\\'
# e_names = read_datas(e_path)[0]
#
#
# a_path_1 ='H:\\青藏高原数据\\时间预测\\2015-2017\\result(不分块)\\随机部分日期数据\\gauge\\' #改路径
# b_path_1= 'H:\\青藏高原数据\\时间预测\\2015-2017\\result(不分块)\\随机部分日期数据\\国家气象局\\'  #改路径
# c_path_1 = 'H:\\青藏高原数据\\时间预测\\2015-2017\\01_clip_data\\xunlian\\国家气象局\\' #改路径
# d_path_1 = 'H:\\青藏高原数据\\时间预测\\2015-2017\\01_clip_data\\xunlian_features\\lat_wei\\' #改路径
# e_path_1 = 'H:\\青藏高原数据\\时间预测\\2015-2017\\01_clip_data\\xunlian_features\\wendu\\' #改路径
#
# a_names_1 = read_datas(a_path_1)[0]
# b_names_1 = read_datas(b_path_1)[0]
# c_names_1 = read_datas(c_path_1)[0]
# d_names_1 = read_datas(d_path_1)[0]
# e_names_1 = read_datas(e_path_1)[0]
#
#
# a_path_2 = 'F:\\青藏高原\\空间\\年度\\6-8\\gsmap_mvk\\' #改路径
# b_path_2= 'F:\\青藏高原\\空间\\年度\\6-8\\gsmap_gauge\\'
# c_path_2 = 'F:\\青藏高原\\空间\\年度\\6-8\\国家气象局\\'
# d_path_2 = 'F:\\青藏高原\\空间\\年度\\6-8\\ANN\\' #改路径
# # e_path_2 = 'F:\\空间校正结果\\' + climate +'\\'+ banqiu +'\\有DNN\\'+ season2 +'\\ANN\\' #改路径
#
# a_names_2 = read_datas(a_path_2)[0]
# b_names_2 = read_datas(b_path_2)[0]
# c_names_2 = read_datas(c_path_2)[0]
# d_names_2 = read_datas(d_path_2)[0]
# # e_names_2 = read_datas(e_path_2)[0]
#
# a_path_3 = 'F:\\青藏高原\\空间\\年度\\9-11\\gsmap_mvk\\' #改路径
# b_path_3= 'F:\\青藏高原\\空间\\年度\\9-11\\gsmap_gauge\\'
# c_path_3 = 'F:\\青藏高原\\空间\\年度\\9-11\\国家气象局\\'
# d_path_3 = 'F:\\青藏高原\\空间\\年度\\9-11\\ANN\\' #改路径
# # e_path_3 = 'F:\\空间校正结果\\' + climate +'\\'+ banqiu +'\\有DNN\\'+ season3 +'\\ANN\\' #改路径
#
# a_names_3 = read_datas(a_path_3)[0]
# b_names_3 = read_datas(b_path_3)[0]
# c_names_3 = read_datas(c_path_3)[0]
# d_names_3 = read_datas(d_path_3)[0]
# # e_names_3 = read_datas(e_path_3)[0]
#
# a_path_4 = 'F:\\青藏高原\\空间\\年度\\12-2\\gsmap_mvk\\' #改路径
# b_path_4= 'F:\\青藏高原\\空间\\年度\\12-2\\gsmap_gauge\\'
# c_path_4 = 'F:\\青藏高原\\空间\\年度\\12-2\\国家气象局\\'
# d_path_4 = 'F:\\青藏高原\\空间\\年度\\12-2\\ANN\\' #改路径
# # e_path_4 = 'F:\\空间校正结果\\' + climate +'\\'+ banqiu +'\\有DNN\\'+ season4 +'\\ANN\\' #改路径
#
# a_names_4 = read_datas(a_path_4)[0]
# b_names_4 = read_datas(b_path_4)[0]
# c_names_4 = read_datas(c_path_4)[0]
# d_names_4 = read_datas(d_path_4)[0]
# # e_names_4 = read_datas(e_path_4)[0]
#
# for i in range(len(a_names_1)):
#     if a_names_1[i] in e_names:
#         continue
#     else:
#         remove_path = a_path_1 + a_names_1[i]   #改路径
#         os.remove(remove_path)
#
#     if b_names_1[i] in e_names:
#         continue
#     else:
#         remove_path = b_path_1 + b_names_1[i]   #改路径
#         os.remove(remove_path)



# for i in range(len(a_names_2)):
#     if a_names_2[i] in e_names:
#         continue
#     else:
#         remove_path = a_path_2 + a_names_2[i]  # 改路径
#         os.remove(remove_path)
#
#     if b_names_2[i] in e_names:
#         continue
#     else:
#         remove_path = b_path_2 + b_names_2[i]  # 改路径
#         os.remove(remove_path)
#
#     if c_names_2[i] in e_names:
#         continue
#     else:
#         remove_path = c_path_2 + c_names_2[i]  # 改路径
#         os.remove(remove_path)
#
#     if d_names_2[i] in e_names:
#         continue
#     else:
#         remove_path = d_path_2 + d_names_2[i]  # 改路径
#         os.remove(remove_path)
#
#
#
# for i in range(len(a_names_3)):
#     if a_names_3[i] in e_names:
#         continue
#     else:
#         remove_path = a_path_3 + a_names_3[i]  # 改路径
#         os.remove(remove_path)
#
#     if b_names_3[i] in e_names:
#         continue
#     else:
#         remove_path = b_path_3 + b_names_3[i]  # 改路径
#         os.remove(remove_path)
#
#     if c_names_3[i] in e_names:
#         continue
#     else:
#         remove_path = c_path_3 + c_names_3[i]  # 改路径
#         os.remove(remove_path)
#
#     if d_names_3[i] in e_names:
#         continue
#     else:
#         remove_path = d_path_3 + d_names_3[i]  # 改路径
#         os.remove(remove_path)
#
#
#
# for i in range(len(a_names_4)):
#     if a_names_4[i] in e_names:
#         continue
#     else:
#         remove_path = a_path_4 + a_names_4[i]  # 改路径
#         os.remove(remove_path)
#
#     if b_names_4[i] in e_names:
#         continue
#     else:
#         remove_path = b_path_4 + b_names_4[i]  # 改路径
#         os.remove(remove_path)
#
#     if c_names_4[i] in e_names:
#         continue
#     else:
#         remove_path = c_path_4 + c_names_4[i]  # 改路径
#         os.remove(remove_path)
#
#     if d_names_4[i] in e_names:
#         continue
#     else:
#         remove_path = d_path_4 + d_names_4[i]  # 改路径
#         os.remove(remove_path)
#
#


# 文件复制
# # seasons = ['3-5','6-8','9-11','12-2']
# # for season1 in seasons:
#
# a_path = 'G:\\青藏高原\\纬度\\'
# b_path = 'G:\\青藏高原\\温度\\'
# # c_path = 'H:\\青藏高原数据\\国家气象局\\'
# # biaozhun_path = 'F:\\青藏高原\\空间\\1-12\\ANN\\'
#
#
# a_names = read_datas(a_path)[0]
# b_names = read_datas(b_path)[0]
# # c_names = read_datas(c_path)[0]
# # biaozhun_names = read_datas(biaozhun_path)[0]
#
# # file_dir_a = 'F:\\青藏高原\\空间\\1-12\\gsmap_mvk\\'
# # mkdir(file_dir_a)
# # file_dir_b = 'F:\\青藏高原\\空间\\1-12\\gsmap_gauge\\'
# # mkdir(file_dir_b)
# # file_dir_c = 'F:\\青藏高原\\空间\\1-12\\国家气象局\\'
# # mkdir(file_dir_c)
#
# # for i in range(len(a_names)):
# #     if a_names[i] in biaozhun_names:
# #          file = a_path + a_names[i]
# #          shutil.copy(file,file_dir_a)
# #
# # for i in range(len(b_names)):
# #     if b_names[i] in biaozhun_names:
# #          file = b_path + b_names[i]
# #          shutil.copy(file,file_dir_b)
# #
# # for i in range(len(c_names)):
# #     if c_names[i] in biaozhun_names:
# #          file = c_path + c_names[i]
# #          shutil.copy(file,file_dir_c)
#
#
# for i in range(len(a_names)):
#     if a_names[i] in b_names:
#         continue
#     else:
#         print(a_names[i])

