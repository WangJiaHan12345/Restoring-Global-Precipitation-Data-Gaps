import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from matplotlib import cm
import pandas as pd


# X,y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)
# plt.scatter(X[:,0], X[:,1], c='green', marker='o', s=50)
# plt.grid()
# plt.show()
#================================================================================================================#
df_dem = pd.read_excel('D:/水文资料/result_create.xls', sheet_name='DEM')
df_rain = pd.read_excel('D:/水文资料/result_create.xls', sheet_name='Rain')
#================================================================================================================#
df_dem = df_dem.fillna(0)
df_rain = df_rain.fillna(0)
dem_data, rain_data = [], []
for rows in range(df_dem.shape[0]):
    for dem_val, rain_val in zip(df_dem.iloc[rows].values, df_rain.iloc[rows].values):
        if dem_val != 0 and rain_val != 0:
# ================================================================================================================#
            dem_data.append(abs(dem_val) )
            # dem_data.append(abs(dem_val) / 100)
            rain_data.append(abs(rain_val) * 1000)
            # rain_data.append(abs(rain_val))
#================================================================================================================#
print(dem_data)
print(rain_data)
print(len(dem_data))
print(len(rain_data))

n = len(dem_data)
X = np.array([[0] * 2] * n)
X[:,0] = dem_data
X[:,1] = rain_data
plt.scatter(X[:,0], X[:,1], c='green', marker='o', s=50)
plt.grid()
plt.show()
#================================================================================================================#
#Kmeans
km = KMeans(n_clusters=6, init='random', n_init=10, max_iter=300, tol = 1e-04, random_state=0)
y_km = km.fit_predict(X)
plt.scatter(X[y_km==0, 0],X[y_km==0, 1], s=30, c='b', marker='o', label='cluster 1')
plt.scatter(X[y_km==1, 0],X[y_km==1, 1], s=30, c='y', marker='o', label='cluster 2')
plt.scatter(X[y_km==2, 0],X[y_km==2, 1], s=30, c='g', marker='o', label='cluster 3')
# plt.scatter(X[y_km==3, 0],X[y_km==3, 1], s=30, c='lightgreen', marker='o', label='cluster 4')
# plt.scatter(X[y_km==4, 0],X[y_km==4, 1], s=30, c='orange', marker='o', label='cluster 5')
# plt.scatter(X[y_km==5, 0],X[y_km==5, 1], s=30, c='pink', marker='o', label='cluster 6')
#================================================================================================================#

print(y_km)

#================================================================================================================#
# count_0, count_1, count_2 = 0, 0, 0
# # count_0, count_1, count_2, count_3 = 0, 0, 0, 0
# for i in y_km:
#     if i == 0:
#         count_0 += 1
#     elif i ==1:
#         count_1 += 1
#     # elif i == 2:
#     else:
#         count_2 += 1
#     # else:
#     #     count_3 += 1
# print(count_0)
# print(count_1)
# print(count_2)
# # print(count_3)
# # 将标签写入txt文件
# print(len(y_km))
# t = open('E:/LinJN_bishe/暴雨数据/00_DEM_Rain/class.txt','w')
# # t = open('E:/LinJN_bishe/暴雨数据/03_MRDEM_Rain/04/class1.txt','w')
# for i in range(len(y_km)):
#         t.write(str(y_km[i]) + '\n')
# t.close()
#================================================================================================================#

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=250, marker='*', c='orange', label='centroids')
plt.legend()
# plt.ylim([400, 1800])
plt.grid()
plt.show()
print('Distortion: %.2f' %km.inertia_)
#使用肘方法确定最佳簇数量
distortion = []
for i in range(1,11):
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, n_jobs=-1, tol=1e-04, random_state=0)
    km.fit(X)
    distortion.append(km.inertia_)
plt.plot(range(1,11),distortion,marker='o', c = 'black')
plt.xlabel('Number of cluster (k)')
plt.ylabel('Distortion')
# plt.axis([1,10,0,1.6*1e8])
plt.show()
#================================================================================================================#
# #通过轮廓图定量分析聚类质量
# km = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
# #================================================================================================================#
# y_km = km.fit_predict(X)
# cluster_labels = np.unique(y_km)
# print(cluster_labels)
# n_cluster = cluster_labels.shape[0]
# silhouette_val = silhouette_samples(X, y_km, metric='euclidean')
# y_ax_lower, y_ax_upper = 0, 0
# y_ticks = []
# for i, c in enumerate(cluster_labels):
#     c_sihouette_val = silhouette_val[y_km==c]
#     c_sihouette_val.sort()
#     y_ax_upper += len(c_sihouette_val)
#     color = cm.jet(i / n_cluster)
#     plt.barh(range(y_ax_lower,y_ax_upper),c_sihouette_val,height=1.0,edgecolor='none',color=color)
#     y_ticks.append((y_ax_lower+y_ax_upper)/2)
#     y_ax_lower += len(c_sihouette_val)
# silhouette_avg = np.mean(silhouette_val)
# print("silhouette_avg_3:",silhouette_avg)
# plt.axvline(silhouette_avg, color='red', ls='--')
# plt.yticks(y_ticks, cluster_labels+1)
# plt.ylabel('Cluster')
# plt.xlabel('Silhouette coefficient')
# plt.show()
#
#
# #效果不佳轮廓图的形状,簇数量为2
# km2 = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
# y_km2 = km2.fit_predict(X)
# cluster_labels = np.unique(y_km2)
# print(cluster_labels)
# n_cluster = cluster_labels.shape[0]
# silhouette_val = silhouette_samples(X, y_km2, metric='euclidean')
# y_ax_lower, y_ax_upper = 0, 0
# y_ticks = []
# for i, c in enumerate(cluster_labels):
#     c_sihouette_val = silhouette_val[y_km2==c]
#     c_sihouette_val.sort()
#     y_ax_upper += len(c_sihouette_val)
#     color = cm.jet(i / n_cluster)
#     plt.barh(range(y_ax_lower,y_ax_upper),c_sihouette_val,height=1.0,edgecolor='none',color=color)
#     y_ticks.append((y_ax_lower+y_ax_upper)/2)
#     y_ax_lower += len(c_sihouette_val)
# silhouette_avg = np.mean(silhouette_val)
# print("silhouette_avg_2:",silhouette_avg)
# plt.axvline(silhouette_avg, color='red', ls='--')
# plt.yticks(y_ticks, cluster_labels+1)
# plt.ylabel('Cluster')
# plt.xlabel('Silhouette coefficient')
# plt.show()
#
# #效果不佳轮廓图的形状,簇数量为4
# km4 = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
# y_km4 = km4.fit_predict(X)
# cluster_labels = np.unique(y_km4)
# print(cluster_labels)
# n_cluster = cluster_labels.shape[0]
# silhouette_val = silhouette_samples(X, y_km4, metric='euclidean')
# y_ax_lower, y_ax_upper = 0, 0
# y_ticks = []
# for i, c in enumerate(cluster_labels):
#     c_sihouette_val = silhouette_val[y_km4==c]
#     c_sihouette_val.sort()
#     y_ax_upper += len(c_sihouette_val)
#     color = cm.jet(i / n_cluster)
#     plt.barh(range(y_ax_lower,y_ax_upper),c_sihouette_val,height=1.0,edgecolor='none',color=color)
#     y_ticks.append((y_ax_lower+y_ax_upper)/2)
#     y_ax_lower += len(c_sihouette_val)
# silhouette_avg = np.mean(silhouette_val)
# print("silhouette_avg_4:",silhouette_avg)
# plt.axvline(silhouette_avg, color='red', ls='--')
# plt.yticks(y_ticks, cluster_labels+1)
# plt.ylabel('Cluster')
# plt.xlabel('Silhouette coefficient')
# plt.show()
#
# #效果不佳轮廓图的形状,簇数量为5
# km2 = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
# y_km2 = km2.fit_predict(X)
# cluster_labels = np.unique(y_km2)
# print(cluster_labels)
# n_cluster = cluster_labels.shape[0]
# silhouette_val = silhouette_samples(X, y_km2, metric='euclidean')
# y_ax_lower, y_ax_upper = 0, 0
# y_ticks = []
# for i, c in enumerate(cluster_labels):
#     c_sihouette_val = silhouette_val[y_km2==c]
#     c_sihouette_val.sort()
#     y_ax_upper += len(c_sihouette_val)
#     color = cm.jet(i / n_cluster)
#     plt.barh(range(y_ax_lower,y_ax_upper),c_sihouette_val,height=1.0,edgecolor='none',color=color)
#     y_ticks.append((y_ax_lower+y_ax_upper)/2)
#     y_ax_lower += len(c_sihouette_val)
# silhouette_avg = np.mean(silhouette_val)
# print("silhouette_avg_5:",silhouette_avg)
# plt.axvline(silhouette_avg, color='red', ls='--')
# plt.yticks(y_ticks, cluster_labels+1)
# plt.ylabel('Cluster')
# plt.xlabel('Silhouette coefficient')
# plt.show()
#
# #效果不佳轮廓图的形状,簇数量为6
# km2 = KMeans(n_clusters=6, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
# y_km2 = km2.fit_predict(X)
# cluster_labels = np.unique(y_km2)
# print(cluster_labels)
# n_cluster = cluster_labels.shape[0]
# silhouette_val = silhouette_samples(X, y_km2, metric='euclidean')
# y_ax_lower, y_ax_upper = 0, 0
# y_ticks = []
# for i, c in enumerate(cluster_labels):
#     c_sihouette_val = silhouette_val[y_km2==c]
#     c_sihouette_val.sort()
#     y_ax_upper += len(c_sihouette_val)
#     color = cm.jet(i / n_cluster)
#     plt.barh(range(y_ax_lower,y_ax_upper),c_sihouette_val,height=1.0,edgecolor='none',color=color)
#     y_ticks.append((y_ax_lower+y_ax_upper)/2)
#     y_ax_lower += len(c_sihouette_val)
# silhouette_avg = np.mean(silhouette_val)
# print("silhouette_avg_6:",silhouette_avg)
# plt.axvline(silhouette_avg, color='red', ls='--')
# plt.yticks(y_ticks, cluster_labels+1)
# plt.ylabel('Cluster')
# plt.xlabel('Silhouette coefficient')
# plt.show()
#
# #效果不佳轮廓图的形状,簇数量为7
# km2 = KMeans(n_clusters=7, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
# y_km2 = km2.fit_predict(X)
# cluster_labels = np.unique(y_km2)
# print(cluster_labels)
# n_cluster = cluster_labels.shape[0]
# silhouette_val = silhouette_samples(X, y_km2, metric='euclidean')
# y_ax_lower, y_ax_upper = 0, 0
# y_ticks = []
# for i, c in enumerate(cluster_labels):
#     c_sihouette_val = silhouette_val[y_km2==c]
#     c_sihouette_val.sort()
#     y_ax_upper += len(c_sihouette_val)
#     color = cm.jet(i / n_cluster)
#     plt.barh(range(y_ax_lower,y_ax_upper),c_sihouette_val,height=1.0,edgecolor='none',color=color)
#     y_ticks.append((y_ax_lower+y_ax_upper)/2)
#     y_ax_lower += len(c_sihouette_val)
# silhouette_avg = np.mean(silhouette_val)
# print("silhouette_avg_7:",silhouette_avg)
# plt.axvline(silhouette_avg, color='red', ls='--')
# plt.yticks(y_ticks, cluster_labels+1)
# plt.ylabel('Cluster')
# plt.xlabel('Silhouette coefficient')
# plt.show()
#
# #效果不佳轮廓图的形状,簇数量为8
# km2 = KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
# y_km2 = km2.fit_predict(X)
# cluster_labels = np.unique(y_km2)
# print(cluster_labels)
# n_cluster = cluster_labels.shape[0]
# silhouette_val = silhouette_samples(X, y_km2, metric='euclidean')
# y_ax_lower, y_ax_upper = 0, 0
# y_ticks = []
# for i, c in enumerate(cluster_labels):
#     c_sihouette_val = silhouette_val[y_km2==c]
#     c_sihouette_val.sort()
#     y_ax_upper += len(c_sihouette_val)
#     color = cm.jet(i / n_cluster)
#     plt.barh(range(y_ax_lower,y_ax_upper),c_sihouette_val,height=1.0,edgecolor='none',color=color)
#     y_ticks.append((y_ax_lower+y_ax_upper)/2)
#     y_ax_lower += len(c_sihouette_val)
# silhouette_avg = np.mean(silhouette_val)
# print("silhouette_avg_8:",silhouette_avg)
# plt.axvline(silhouette_avg, color='red', ls='--')
# plt.yticks(y_ticks, cluster_labels+1)
# plt.ylabel('Cluster')
# plt.xlabel('Silhouette coefficient')
# plt.show()
#
# #效果不佳轮廓图的形状,簇数量为9
# km2 = KMeans(n_clusters=9, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
# y_km2 = km2.fit_predict(X)
# cluster_labels = np.unique(y_km2)
# print(cluster_labels)
# n_cluster = cluster_labels.shape[0]
# silhouette_val = silhouette_samples(X, y_km2, metric='euclidean')
# y_ax_lower, y_ax_upper = 0, 0
# y_ticks = []
# for i, c in enumerate(cluster_labels):
#     c_sihouette_val = silhouette_val[y_km2==c]
#     c_sihouette_val.sort()
#     y_ax_upper += len(c_sihouette_val)
#     color = cm.jet(i / n_cluster)
#     plt.barh(range(y_ax_lower,y_ax_upper),c_sihouette_val,height=1.0,edgecolor='none',color=color)
#     y_ticks.append((y_ax_lower+y_ax_upper)/2)
#     y_ax_lower += len(c_sihouette_val)
# silhouette_avg = np.mean(silhouette_val)
# print("silhouette_avg_9:",silhouette_avg)
# plt.axvline(silhouette_avg, color='red', ls='--')
# plt.yticks(y_ticks, cluster_labels+1)
# plt.ylabel('Cluster')
# plt.xlabel('Silhouette coefficient')
# plt.show()
#
# #效果不佳轮廓图的形状,簇数量为10
# km2 = KMeans(n_clusters=10, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
# y_km2 = km2.fit_predict(X)
# cluster_labels = np.unique(y_km2)
# print(cluster_labels)
# n_cluster = cluster_labels.shape[0]
# silhouette_val = silhouette_samples(X, y_km2, metric='euclidean')
# y_ax_lower, y_ax_upper = 0, 0
# y_ticks = []
# for i, c in enumerate(cluster_labels):
#     c_sihouette_val = silhouette_val[y_km2==c]
#     c_sihouette_val.sort()
#     y_ax_upper += len(c_sihouette_val)
#     color = cm.jet(i / n_cluster)
#     plt.barh(range(y_ax_lower,y_ax_upper),c_sihouette_val,height=1.0,edgecolor='none',color=color)
#     y_ticks.append((y_ax_lower+y_ax_upper)/2)
#     y_ax_lower += len(c_sihouette_val)
# silhouette_avg = np.mean(silhouette_val)
# print("silhouette_avg_10:",silhouette_avg)
# plt.axvline(silhouette_avg, color='red', ls='--')
# plt.yticks(y_ticks, cluster_labels+1)
# plt.ylabel('Cluster')
# plt.xlabel('Silhouette coefficient')
# plt.show()