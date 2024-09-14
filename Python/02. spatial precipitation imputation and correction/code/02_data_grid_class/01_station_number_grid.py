# # -*- coding: utf-8 -*-
# """
# 将年鉴数据里面雨量站点放入对应网格里面，每个网格雨量站点计数
# """
# import numpy as np
# import pandas as pd
#
# df = pd.read_csv("F:\\水文资料\\nenjianglianxi\\2016嫩江区.csv")
# data_ = np.array(df)
# data = data_[:, 0:2]  # 所有行的第一列和第二列  故此时data只是经纬度数据
# # print(data)
# rows = data.shape[0]     #rows代表行 cols代表列
# cols = data.shape[1]
#
# station = np.zeros((70,85)).astype('int64')  # 共70行 每行85列 初值为0
# for row in range(rows):
#     lon = data[row, 0]  # 101.85
#     lat = data[row, 1]  # 26.6
#     # print(lon, lat)
#     for i in range(70):
#         lat_r = 51.40833333 - 0.1 * i  # 44.40833333+7  因为起点在左下角
#         lat_l = lat_r - 0.1
#         if lat_l <= lat < lat_r:
#             for j in range(85):
#                 lon_l = 119.405 + 0.1 * j
#                 lon_r = lon_l + 0.1
#                 if lon_l <= lon < lon_r:
#                     # print(station[i, j])
#                     # print('*' * 10)
#                     station[i, j] += 1
#                     # print(station[i, j])
#                     break
#         else:
#             continue
#         break
# # print(station)
# data1 = pd.DataFrame(station)   # DataFrame通常用于二维数据的输入 得到类似excle表格的数据格式
# data1.to_csv('F:\\shuiwenziliao\\nenjianglianxi\\station.csv')



# #excle  txt
# import pandas as pd
#
# df = pd.read_excel('H:\\5\\class_DEM_sumrain.xlsx', sheet_name='class_DEM_sumrain', header=None)		# 使用pandas模块读取数据
# print('开始写入txt文件...')
# df.to_csv('H:\\5\\class_DEM_sumrain1.txt', header=None, sep=' ', index=False)		# 写入，逗号分隔
# print('文件写入成功!')

# import numpy as np
#
#
# data = np.fromfile('E:\\1\\SEVP_CLI_CHN_MERGE_CMP_PRE_HOUR_GRID_0.10-2018010100.grd', dtype="<f4")
# print(data)
# data = data.reshape((880, 700), order='C')[::-1]
# print(data)
# gsamp = data[:440]
# crain = data[440:]
# print(gsamp)
# print(crain)
#
# file3 = open(r'E:\\1\\' +'1.txt', 'w')
# for i in range (len(crain)):
#     file3.write(str(crain[i])+'\n')
# file3.close()
# #     filename ='E:\\1\\SEVP_CLI_CHN_MERGE_CMP_PRE_HOUR_GRID_0.10-2015010100.grd'
# #     a,b=read(filename)
# #     print(a,b)
