# -*- coding: utf-8 -*-
"""
using csv, not pandas
https://www.kaggle.com/c/titanic/details/
getting-started-with-python-ii
"""

import csv as csv
import numpy as np

csv_file_object = csv.reader(open('data/train.csv','rb'))
header = csv_file_object.next()
data=[]

for row in csv_file_object:
    data.append(row)
data = np.array(data)

print data[0:15,5]
print type(data[0::,5])


'''
年齢の平均値を算出するため、
pandasを使って、欠損値を埋め、stringをfloatに変換する
'''

