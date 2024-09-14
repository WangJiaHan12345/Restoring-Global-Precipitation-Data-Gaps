import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from matplotlib import cm
import pandas as pd
import matplotlib
import os
import math
from mpl_toolkits.mplot3d import Axes3D

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


def read_csv(path_csv):
    df = pd.read_csv(path_csv)
    data = np.array(df)
    return data


path_rain_csv = "H:\\时间预测\\四个区域可用数据\\shirun\\rain\\rain_sum_12-2_n.csv"
data_r = read_csv(path_rain_csv)
#北半球n
# data_rain = data_r[:120, 1:]
# 南半球
data_rain = data_r[120:, 1:]


path_dem_csv = "H:\\时间预测\\四个区域可用数据\\shirun\\dem\\DEM.csv"
data_d = read_csv(path_dem_csv)
#北半球
# data_dem = data_d[:120, 1:]
# 南半球
data_dem = data_d[120:, 1:]

#
path_mrdem_csv = "H:\\时间预测\\四个区域可用数据\\shirun\\dem\\MRDEM.csv"
data_mrd = read_csv(path_mrdem_csv)
#北半球
# data_mrdem = data_mrd[:120, 1:]
# 南半球
data_mrdem = data_mrd[120:, 1:]


rows = data_dem.shape[0]
cols = data_dem.shape[1]
print(rows)
print(cols)

# 北半球
# dem_data, rain_data, mrdem_data = [], [], []
# for row in range(rows):
#     for col in range(cols):
#         if data_dem[row, col] != -9999.0 and data_rain[row, col] != -9999.0 and data_mrdem[row, col] != -9999.0 :
#             dem_data.append(data_dem[row, col])
#             rain_data.append(data_rain[row, col])
#             mrdem_data.append(data_mrdem[row, col])

# #南半球
dem_data, rain_data, mrdem_data = [], [], []
for row in range(rows):
    for col in range(cols):
        if data_dem[row, col] != -9999.0 and data_rain[row, col] != -9999.0 and data_mrdem[row, col] != -9999.0 :
            dem_data.append(data_dem[row, col])
            rain_data.append(data_rain[row, col])
            mrdem_data.append(data_mrdem[row, col])



n = len(dem_data)
X = np.array([[0] * 3] * n) #N行两列
X[:, 0] = dem_data
X[:, 1] = mrdem_data
X[:, 2] = rain_data


fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1],X[:, 2],s=10,c='green', marker='o')
plt.grid()
plt.show()
# ================================================================================================================#
# Kmeans
km = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
y_km = km.fit_predict(X) #表示拟合加上预测
fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[y_km==0, 0], X[y_km==0, 1], X[y_km==0, 2],s=10,  c='b', marker='o', label='cluster 1')
ax.scatter(X[y_km==1, 0], X[y_km==1, 1], X[y_km==1, 2],s=10,  c='y', marker='o', label='cluster 2')
ax.scatter(X[y_km==2, 0], X[y_km==2, 1], X[y_km==2, 2],s=10,  c='g', marker='o', label='cluster 3')
ax.scatter(X[y_km==3, 0], X[y_km==3, 1], X[y_km==3, 2],s=10,  c='lightgreen', marker='o', label='cluster 4')
# ax.scatter(X[y_km==4, 0], X[y_km==4, 1], X[y_km==4, 2],s=10,  c='orange', marker='o', label='cluster 5')
# ax.scatter(X[y_km==5, 0], X[y_km==5, 1], X[y_km==5, 2],s=10,  c='pink', marker='o', label='cluster 6')
# ax.scatter(X[y_km==6, 0], X[y_km==6, 1], X[y_km==6, 2],s=10,  c='purple', marker='o', label='cluster 7')
# ax.scatter(X[y_km==7, 0], X[y_km==7, 1], X[y_km==7, 2],s=10,  c='purple', marker='o', label='cluster 8')
#= ===============================================================================================================#
# plt.show()
print(y_km)

# ================================================================================================================#
# count_0, count_1 = 0, 0
count_0, count_1, count_2, count_3, count_4 = 0, 0, 0, 0, 0
for i in y_km:
    if i == 0:
        count_0 += 1
    elif i == 1:
        count_1 += 1
    elif i == 2:
        count_2 += 1
    elif i == 3:
        count_3 += 1
    # elif i == 4:
    #     count_4 += 1
    # elif i == 5:
    #     count_5 += 1
    # else:
    #     count_6 += 1
print('count_0',count_0)
print('count_1',count_1)
print('count_2',count_2)
print('count_3',count_3)
# print('count_4',count_4)
# print('count_5',count_5)
# print('count_6',count_6)
# print(count_6)
# print(count_7)
# 将标签写入txt文件
print(len(y_km))

t = open('H:\\时间预测\\四个区域可用数据\\shirun\\class_sumrain_12-2_n_shiyan.txt', 'w') #得到每个点是属于第几个聚类
for i in range(len(y_km)):
    t.write(str(y_km[i]) + '\n')
t.close()
# ================================================================================================================#

#每次簇的中心点
ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], km.cluster_centers_[:, 2], s=30, marker='*', c='orange', label='centroids')
plt.legend() #给图像加图例
# plt.ylim([400, 1800])
plt.grid() #给图像加网格线
plt.show()
print('Distortion: %.2f' % km.inertia_)

x = np.linspace(0, 1, 25)

# 使用肘方法确定最佳簇数量
distortion = []  #失真
for i in range(1, 26):
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, n_jobs=-1, tol=1e-04, random_state=0)
    km.fit(X)
    distortion.append(km.inertia_)#误差平方和，每个点到聚类中心点的距离之和
print(distortion)



fig=plt.figure()
ax=fig.add_subplot()
ax.plot(range(1, 26), distortion, marker='o', c='black')

plt.plot(x,x+2,'k.-',label='SSE')
plt.plot(x,x**2,'r^-',label='SC')


plt.xlabel('(d) Class Number k',fontsize=23)
plt.ylabel('SSE',fontsize=24)
plt.legend(prop={'size':20})

#需要自己指定数字
plt.axvline(4,color="black",ls="--")

ax.tick_params(labelsize=20)
ax.set_ylim([0,10*1e9])
ax.set_xlim([0,26])


# ================================================================================================================#
# 通过轮廓图定量分析聚类质量
silhouette_avg=[]
for i in range(2,26):
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    # ================================================================================================================#
    y_km = km.fit_predict(X)
    cluster_labels = np.unique(y_km)
    print(cluster_labels)
    n_cluster = cluster_labels.shape[0]
    silhouette_val = silhouette_samples(X, y_km, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    y_ticks = []
    for i, c in enumerate(cluster_labels):
        c_sihouette_val = silhouette_val[y_km==c]
        c_sihouette_val.sort()
        y_ax_upper += len(c_sihouette_val)
        color = cm.jet(i / n_cluster)
        plt.barh(range(y_ax_lower,y_ax_upper),c_sihouette_val,height=1.0,edgecolor='none',color=color)
        y_ticks.append((y_ax_lower+y_ax_upper)/2)
        y_ax_lower += len(c_sihouette_val)
    avg = np.mean(silhouette_val)
    # print("silhouette_avg",avg)
    print("avg",avg)
    # plt.axvline(avg,color="red",ls="--")
    # plt.yticks(y_ticks,cluster_labels+1)
    silhouette_avg.append(avg)


ax2=ax.twinx()
ax2.set_ylim(0,1)
ax2.plot(range(2, 26), silhouette_avg, marker='^', c='red')
ax2.set_ylabel('SC',fontsize=24)
#刻度值字体大小设置（x轴和y轴同时设置）
ax2.tick_params(labelsize=20)


# plt.yticks([])
plt.savefig("H:\\时间预测\\四个区域可用数据\\shirun\\figures\\南\\12-2_shiyan.png", bbox_inches='tight')
plt.show()



