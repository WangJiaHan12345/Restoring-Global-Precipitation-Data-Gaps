import pandas as pd

df = pd.read_excel('D:\\03Paper\\2017Beijiang\\data\\2017北江区_1.xlsx', header=None)		# 使用pandas模块读取数据
print('开始写入txt文件...')
df.to_csv('D:\\03Paper\\2017Beijiang\\data\\2017北江区_4.csv', header=None, sep=',', index=False)		# 写入，逗号分隔
print('文件写入成功!')
