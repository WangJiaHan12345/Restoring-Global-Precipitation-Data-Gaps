import scipy.stats as stats
import scipy
# x=[10.35,6.24,3.18,8.46,3.21,7.65,4.32,8.66,9.12,10.31]
# y=[5.1,3.15,1.67,4.33,1.76,4.11,2.11,4.88,4.99,5.12]


file_txt = open('G:\\全球\\时间预测结果\\画相对图所用数据\\季度\\原始数据_1\\分气候区\\湿润区\\经纬度相关性\\12-2.txt')
lines = file_txt.readlines()


x = []
for line in lines:
    x.append(float(line.strip('\n')))


file_txt1 = open('G:\\全球\\时间预测结果\\画相对图所用数据\\季度\\原始数据_1\\分气候区\\湿润区\\经纬度相关性\\lon.txt')
lines1 = file_txt1.readlines()

y = []
for line in lines1:
    y.append(float(line.strip('\n')))


correlation,pvalue=stats.stats.pearsonr(x,y)
print('correlation:',correlation)
print('pvalue:',pvalue)