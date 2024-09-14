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


# 文件移除
# climate = 'ganhan'
# banqiu = 'global_n+s'
# kind = 'gauge4'
# season1 = '3-5'
# season2 = '6-8'
# season3 = '9-11'
# season4 = '12-2'
# block = 'A'

# 改路径
a_path = 'H:\\青藏高原数据\\空间+时间\\01_data\\2015-2016\\国家气象局\\3-5\\' #改路径
b_path = 'H:\\青藏高原数据\\空间+时间\\01_data\\2015-2016\\国家气象局\\6-8\\'
c_path = 'H:\\青藏高原数据\\空间+时间\\01_data\\2015-2016\\国家气象局\\9-11\\'
d_path = 'H:\\青藏高原数据\\空间+时间\\01_data\\2015-2016\\国家气象局\\12-2\\'

train = True

a_names = read_datas(a_path)[0]
b_names = read_datas(b_path)[0]
c_names = read_datas(c_path)[0]
d_names = read_datas(d_path)[0]
# e_names = read_datas(e_path)[0]


# a_path_1 = 'H:\\青藏高原数据\\空间+时间\\01_data\\2015-2016.5\\国家气象局\\3-5\\' #改路径
# b_path_1 = 'H:\\青藏高原数据\\空间+时间\\01_data\\2015-2016.5\\国家气象局\\6-8\\'
# c_path_1 = 'H:\\青藏高原数据\\空间+时间\\01_data\\2015-2016.5\\国家气象局\\9-11\\'
# d_path_1 = 'H:\\青藏高原数据\\空间+时间\\01_data\\2015-2016.5\\国家气象局\\12-2\\'
#
#
#
# a_names_1 = read_datas(a_path_1)[0]
# b_names_1 = read_datas(b_path_1)[0]
# c_names_1 = read_datas(c_path_1)[0]
# d_names_1 = read_datas(d_path_1)[0]


# 在同一个文件夹下进行删除
for i in range(len(a_names)):
    if a_names[i] in b_names and a_names[i] in c_names and a_names[i] in d_names:
        continue
    else:
        remove_path = a_path + a_names[i]   #改路径
        os.remove(remove_path)

for i in range(len(b_names)):
    if b_names[i] in a_names and b_names[i] in c_names and b_names[i] in d_names:
        continue
    else:
        remove_path = b_path + b_names[i]   #改路径
        os.remove(remove_path)

for i in range(len(c_names)):
    if c_names[i] in a_names and c_names[i] in b_names and c_names[i] in d_names:
        continue
    else:
        remove_path = c_path + c_names[i]   #改路径
        os.remove(remove_path)

for i in range(len(d_names)):
    if d_names[i] in a_names and d_names[i] in b_names and d_names[i] in c_names:
        continue
    else:
        remove_path = d_path + d_names[i]   #改路径
        os.remove(remove_path)


# e_path = 'G:\\青藏高原\\时间\\从空间取得的数据\\02_season_data\\xunlian\\ANN\\3-5\\'
# e_names = read_datas(e_path)[0]
#
#
# a_path_1 = 'G:\\青藏高原\\时间\从空间取得的数据\\02_season_data\\ceshi\\dem\\3-5\\'
# b_path_1 =  'G:\\青藏高原\\时间\从空间取得的数据\\02_season_data\\ceshi\\dem\\6-8\\'
# c_path_1 =  'G:\\青藏高原\\时间\从空间取得的数据\\02_season_data\\ceshi\\dem\\9-11\\'
# d_path_1 = 'G:\\青藏高原\\时间\从空间取得的数据\\02_season_data\\ceshi\\dem\\12-2\\'
#
# a_names_1 = read_datas(a_path_1)[0]
# b_names_1 = read_datas(b_path_1)[0]
# c_names_1 = read_datas(c_path_1)[0]
# d_names_1 = read_datas(d_path_1)[0]


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
#
#     if c_names_1[i] in e_names:
#         continue
#     else:
#         remove_path = c_path_1 + c_names_1[i]   #改路径
#         os.remove(remove_path)
#
#     if d_names_1[i] in e_names:
#         continue
#     else:
#         remove_path = d_path_1 + d_names_1[i]   #改路径
#         os.remove(remove_path)



# # for i in range(len(a_names_2)):
# #     if a_names_2[i] in e_names:
# #         continue
# #     else:
# #         remove_path = a_path_2 + a_names_2[i]  # 改路径
# #         os.remove(remove_path)
# #
# #     if b_names_2[i] in e_names:
# #         continue
# #     else:
# #         remove_path = b_path_2 + b_names_2[i]  # 改路径
# #         os.remove(remove_path)
# #
# #     if c_names_2[i] in e_names:
# #         continue
# #     else:
# #         remove_path = c_path_2 + c_names_2[i]  # 改路径
# #         os.remove(remove_path)
# #
# #     if d_names_2[i] in e_names:
# #         continue
# #     else:
# #         remove_path = d_path_2 + d_names_2[i]  # 改路径
# #         os.remove(remove_path)
# #
# #
# #
# # for i in range(len(a_names_3)):
# #     if a_names_3[i] in e_names:
# #         continue
# #     else:
# #         remove_path = a_path_3 + a_names_3[i]  # 改路径
# #         os.remove(remove_path)
# #
# #     if b_names_3[i] in e_names:
# #         continue
# #     else:
# #         remove_path = b_path_3 + b_names_3[i]  # 改路径
# #         os.remove(remove_path)
# #
# #     if c_names_3[i] in e_names:
# #         continue
# #     else:
# #         remove_path = c_path_3 + c_names_3[i]  # 改路径
# #         os.remove(remove_path)
# #
# #     if d_names_3[i] in e_names:
# #         continue
# #     else:
# #         remove_path = d_path_3 + d_names_3[i]  # 改路径
# #         os.remove(remove_path)
# #
# #
# #
# # for i in range(len(a_names_4)):
# #     if a_names_4[i] in e_names:
# #         continue
# #     else:
# #         remove_path = a_path_4 + a_names_4[i]  # 改路径
# #         os.remove(remove_path)
# #
# #     if b_names_4[i] in e_names:
# #         continue
# #     else:
# #         remove_path = b_path_4 + b_names_4[i]  # 改路径
# #         os.remove(remove_path)
# #
# #     if c_names_4[i] in e_names:
# #         continue
# #     else:
# #         remove_path = c_path_4 + c_names_4[i]  # 改路径
# #         os.remove(remove_path)
# #
# #     if d_names_4[i] in e_names:
# #         continue
# #     else:
# #         remove_path = d_path_4 + d_names_4[i]  # 改路径
# #         os.remove(remove_path)




# 文件复制
# seasons = ['A','B','C']
# for season in seasons:
#     rnt_path = 'H:\\青藏高原数据\\时间预测\\2015-2016\\只含地面站点\\01_clip_data\\测试\\gsmap_rnt\\3-5\\'
#     mvk_path = 'H:\\青藏高原数据\\时间预测\\2015-2016\\只含地面站点\\01_clip_data\\测试\\gsmap_rnt\\6-8\\'
#     cpc_path = 'H:\\青藏高原数据\\时间预测\\2015-2016\\只含地面站点\\01_clip_data\\测试\\gsmap_rnt\\9-11\\'
#     dem_path = 'H:\\青藏高原数据\\时间预测\\2015-2016\\只含地面站点\\01_clip_data\\测试\\gsmap_rnt\\12-2\\'
#     # lat_path = 'H:\\青藏高原数据\\时间预测\\2015-2016\\全部网格\\01_clip_data\\测试\\纬度\\'+ season +'\\'
#
#
#     a_path = 'H:\\青藏高原数据\\时间预测\\2015-2016\\只含地面站点\\02_final_data\\3-5\\ceshi\\dem\\' + season +'\\'
#     b_path = 'H:\\青藏高原数据\\时间预测\\2015-2016\\只含地面站点\\02_final_data\\6-8\\ceshi\\dem\\' + season +'\\'
#     c_path = 'H:\\青藏高原数据\\时间预测\\2015-2016\\只含地面站点\\02_final_data\\9-11\\ceshi\\dem\\' + season +'\\'
#     d_path = 'H:\\青藏高原数据\\时间预测\\2015-2016\\只含地面站点\\02_final_data\\12-2\\ceshi\\dem\\' + season +'\\'
#
#
#     a_names = read_datas(a_path)[0]
#     b_names = read_datas(b_path)[0]
#     c_names = read_datas(c_path)[0]
#     d_names = read_datas(d_path)[0]
#
#     file_dir_a = 'H:\\青藏高原数据\\时间预测\\2015-2016\\只含地面站点\\02_final_data\\3-5\\ceshi\\gsmap_rnt\\' + season +'\\'
#     mkdir(file_dir_a)
#     file_dir_a1 = 'H:\\青藏高原数据\\时间预测\\2015-2016\\只含地面站点\\02_final_data\\6-8\\ceshi\\gsmap_rnt\\' + season  +'\\'
#     mkdir(file_dir_a1)
#     file_dir_a2 = 'H:\\青藏高原数据\\时间预测\\2015-2016\\只含地面站点\\02_final_data\\9-11\\ceshi\\gsmap_rnt\\' + season  +'\\'
#     mkdir(file_dir_a2)
#     file_dir_a3 ='H:\\青藏高原数据\\时间预测\\2015-2016\\只含地面站点\\02_final_data\\12-2\\ceshi\\gsmap_rnt\\' + season  +'\\'
#     mkdir(file_dir_a3)
    # file_dir_a4 ='H:\\青藏高原数据\\时间预测\\2015-2016\\只含地面站点\\02_final_data\\3-5\\xunlian\\gsmap_rnt\\'
    # mkdir(file_dir_a4)



    # file_dir_b = 'G:\\毕业论文图\\全球\\时间\\柱状图\\半湿润\\'+ season +'\\cpc\\'
    # mkdir(file_dir_b)
    # file_dir_b1 = 'G:\\毕业论文图\\全球\\时间\\柱状图\\半湿润\\'+ season +'\\early\\'
    # mkdir(file_dir_b1)
    # file_dir_b2 = 'G:\\毕业论文图\\全球\\时间\\柱状图\\半湿润\\'+ season +'\\final\\'
    # mkdir(file_dir_b2)
    # file_dir_b3 = 'G:\\毕业论文图\\全球\\时间\\柱状图\\半湿润\\'+ season +'\\ANN\\'
    # mkdir(file_dir_b3)
    #
    #
    # file_dir_c = 'G:\\毕业论文图\\全球\\时间\\柱状图\\半干旱\\'+ season +'\\cpc\\'
    # mkdir(file_dir_c)
    # file_dir_c1 = 'G:\\毕业论文图\\全球\\时间\\柱状图\\半干旱\\'+ season +'\\early\\'
    # mkdir(file_dir_c1)
    # file_dir_c2 = 'G:\\毕业论文图\\全球\\时间\\柱状图\\半干旱\\'+ season +'\\final\\'
    # mkdir(file_dir_c2)
    # file_dir_c3 = 'G:\\毕业论文图\\全球\\时间\\柱状图\\半干旱\\'+ season +'\\ANN\\'
    # mkdir(file_dir_c3)
    #
    # file_dir_d = 'G:\\毕业论文图\\全球\\时间\\柱状图\\干旱\\'+ season +'\\cpc\\'
    # mkdir(file_dir_d)
    # file_dir_d1 = 'G:\\毕业论文图\\全球\\时间\\柱状图\\干旱\\'+ season +'\\early\\'
    # mkdir(file_dir_d1)
    # file_dir_d2 = 'G:\\毕业论文图\\全球\\时间\\柱状图\\干旱\\'+ season +'\\final\\'
    # mkdir(file_dir_d2)
    # file_dir_d3 = 'G:\\毕业论文图\\全球\\时间\\柱状图\\干旱\\'+ season +'\\ANN\\'
    # mkdir(file_dir_d3)


    # for i in range(len(a_names)):
    #      file = rnt_path + a_names[i]
    #      shutil.copy(file,file_dir_a)
    #
    # for i in range(len(b_names)):
    #      file = mvk_path + b_names[i]
    #      shutil.copy(file,file_dir_a1)
    #
    # for i in range(len(c_names)):
    #      file = cpc_path + c_names[i]
    #      shutil.copy(file,file_dir_a2)
    #
    # for i in range(len(d_names)):
    #      file = dem_path + d_names[i]
    #      shutil.copy(file,file_dir_a3)

    # for i in range(len(a_names)):
    #      file = lat_path + a_names[i]
    #      shutil.copy(file,file_dir_a4)

    # for i in range(len(b_names)):
    #      file = cpc_path + b_names[i]
    #      shutil.copy(file,file_dir_b)
    #
    # for i in range(len(b_names)):
    #      file = early_path + b_names[i]
    #      shutil.copy(file,file_dir_b1)
    #
    # for i in range(len(b_names)):
    #      file = final_path + b_names[i]
    #      shutil.copy(file,file_dir_b2)
    #
    # for i in range(len(b_names)):
    #      file = ANN_path + b_names[i]
    #      shutil.copy(file,file_dir_b3)
    #
    #
    # for i in range(len(c_names)):
    #      file = cpc_path + c_names[i]
    #      shutil.copy(file,file_dir_c)
    #
    # for i in range(len(c_names)):
    #      file = early_path + c_names[i]
    #      shutil.copy(file,file_dir_c1)
    #
    # for i in range(len(c_names)):
    #      file = final_path + c_names[i]
    #      shutil.copy(file,file_dir_c2)
    #
    # for i in range(len(c_names)):
    #      file = ANN_path + c_names[i]
    #      shutil.copy(file,file_dir_c3)
    #
    #
    # for i in range(len(d_names)):
    #      file = cpc_path + d_names[i]
    #      shutil.copy(file,file_dir_d)
    #
    # for i in range(len(d_names)):
    #      file = early_path + d_names[i]
    #      shutil.copy(file,file_dir_d1)
    #
    # for i in range(len(d_names)):
    #      file = final_path + d_names[i]
    #      shutil.copy(file,file_dir_d2)
    #
    # for i in range(len(d_names)):
    #      file = ANN_path + d_names[i]
    #      shutil.copy(file,file_dir_d3)