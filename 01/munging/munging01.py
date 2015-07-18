# -*- coding: utf-8 -*-
"""
using csv, not pandas
pythonのCSVライブラリを使っても、あまりいいことがないよ、
pandas使ってね、という説明をするためだけのコード
munging02に飛びましょう。
https://www.kaggle.com/c/titanic/details/
getting-started-with-python-ii
"""

import csv as csv
import numpy as np

csv_file_object = csv.reader(open('../data/train.csv','rb'))
header = csv_file_object.next()
data=[]

for row in csv_file_object:
    data.append(row)
data = np.array(data)

#データをすべて表示する。→すべてstring型
print data

#年齢(6列目)をすべて表示
print type(data[0::,5])

#年齢の平均値を出すため、年齢のstring型をfloat型にしてみよう　→失敗。
#欠損値があってエラーが出てしまうヨーダ...
ages_onboard = data[0::,5].astype(np.float) 

#pandasを使って、その辺うまくやろう→　munging02に続く...