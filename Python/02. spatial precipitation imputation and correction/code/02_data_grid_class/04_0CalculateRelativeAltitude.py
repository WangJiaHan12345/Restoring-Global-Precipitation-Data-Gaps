# -*- coding: utf-8 -*-
"""
此处，由于计算RDEM等需要使用dem整体外扩一圈-9999的xlsx文件
最后保存csv时，要把多出来是外围两行两列去掉
center - average = RDEM
"""
import xlrd
import xlwt
import pandas as pd
import numpy as np
import os


# 打开文件
workbook = xlrd.open_workbook('H:\\青藏高原数据\\空间\\dem\\dem.xlsx') #xlsx就是excle
# 根据sheet索引或者名称获取sheet内容
sheet = workbook.sheet_by_name('Sheet1')
# sheet的名称，行数，列数
print(sheet.name, sheet.nrows, sheet.ncols)

rows = sheet.nrows - 2
cols = sheet.ncols - 2
print(rows,cols)

sheet_row = []
sheet_col = []
for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        if sheet.cell(i, j).value != -9999:
            sheet_row.append(i)
            sheet_col.append(j)
print(sheet_row, sheet_col)

# print(sheet_row[0]-1)
# a = sheet_row[0]
# b = sheet_col[0]
# print(sheet.cell(a, b).value)
# a1 = sheet.cell(a-1, b-1).value
# a2 = sheet.cell(a-1, b).value
# a3 = sheet.cell(a-1, b+1).value
# a4 = sheet.cell(a, b-1).value
# a6 = sheet.cell(a, b+1).value
# a7 = sheet.cell(a+1, b-1).value
# a8 = sheet.cell(a+1, b).value
# a9 = sheet.cell(a+1, b+1).value
# print(a1, a2, a3, a4, a6, a7, a8, a9)

# 获取单元格的内容
center_new = []  #存放结果
for i in range(len(sheet_row)):
    a = sheet_row[i]
    b = sheet_col[i]
    center = sheet.cell(a, b).value
    print(center)

    a1 = sheet.cell(a-1, b-1).value
    a2 = sheet.cell(a-1, b).value
    a3 = sheet.cell(a-1, b+1).value
    a4 = sheet.cell(a, b-1).value
    a6 = sheet.cell(a, b+1).value
    a7 = sheet.cell(a+1, b-1).value
    a8 = sheet.cell(a+1, b).value
    a9 = sheet.cell(a+1, b+1).value

    if sheet.cell(a-1, b-1).value == -9999:
        # sheet.cell(a-1,b-1).value = center
        a1 = center
        print(sheet.cell(a-1, b-1).value)
        print(a1)
    else:
        print('1')

    if sheet.cell(a-1, b).value == -9999:
        # sheet.cell(a-1,b).value =center
        a2 = center
        print(sheet.cell(a-1, b).value)
        print(a2)
    else:
        print('2')

    if sheet.cell(a-1, b+1).value == -9999:
        # sheet.cell(a-1,b+1).value = center
        a3 = center
        print(sheet.cell(a-1, b-1).value)
        print(a3)
    else:
        print('3')

    if sheet.cell(a, b-1).value == -9999:
        # sheet.cell(a,b-1).value = center
        a4 = center
        print(sheet.cell(a, b-1).value)
        print(a4)
    else:
        print('4')

    if sheet.cell(a, b+1).value == -9999:
        # sheet.cell(a,b+1).value = center
        a6 = center
        print(sheet.cell(a, b+1).value)
        print(a6)
    else:
        print('6')

    if sheet.cell(a+1, b-1).value == -9999:
        # sheet.cell(a+1,b-1).value = center
        a7 = center
        print(sheet.cell(a+1, b-1).value)
        print(a7)
    else:
        print('7')

    if sheet.cell(a+1, b).value == -9999:
        # sheet.cell(a+1,b).value = center
        a8 = center
        print(sheet.cell(a+1, b).value)
        print(a8)
    else:
        print('8')

    if sheet.cell(a+1, b+1).value == -9999:
        # sheet.cell(a+1,b+1).value = center
        a9 = center
        print(sheet.cell(a+1, b+1).value)
        print(a9)
    else:
        print('9')


# 求平均海拔差
#     sum = center + a1 + a2 + a3 + a4 + a6 + a7 + a8 + a9
#     average = sum / 9
#     c = average

# 求海拔差最大值
    c1 = abs(center - a1)
    c2 = abs(center - a2)
    c3 = abs(center - a3)
    c4 = abs(center - a4)
    c6 = abs(center - a6)
    c7 = abs(center - a7)
    c8 = abs(center - a8)
    c9 = abs(center - a9)
    c = max(c1, c2, c3, c4, c6, c7, c8, c9)

    # center_new.append(abs(c))
    center_new.append(c)
print(center_new)
print(len(center_new))


dem_data = np.zeros((rows, cols)).astype('int64')
for i in range(rows):
    for j in range(cols):
        dem_data[i, j] = -9999
print(dem_data)

for i in range(len(sheet_row)):
    dem_data[sheet_row[i]-1, sheet_col[i]-1] = center_new[i]
print(dem_data)


data1 = pd.DataFrame(dem_data)
data1.to_csv('H:\\青藏高原数据\\空间\\dem\\MRDEM.csv')
