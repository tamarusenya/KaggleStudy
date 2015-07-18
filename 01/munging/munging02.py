# -*- coding: utf-8 -*-
"""
using pandas

年齢の平均値を算出するため、
pandasを使って、欠損値を埋め、stringをfloatに変換する

https://www.kaggle.com/c/titanic/details/
getting-started-with-python-ii
"""

import pandas as pd
import numpy as np
import pylab as P

df = pd.read_csv('data/train.csv',header=0)

print df.head(3)

#データタイプはpandas特有のデータフレーム
print type(df)

#csvでは、すべてstringだったが、pandasではその他のデータも扱える
print df.dtypes

#各データの数、データ型を一覧できる
print df.info()

#データの数、平均、標準偏差、最小値、パー千タイル(中央値),最大値を一覧できる
print df.describe()

#特定の条件のデータを抽出する
print df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']]

#年齢の欠損値をすべて抽出する
print df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']]

#年齢のヒストグラムを出してみる
#df['Age'].hist()

#年齢のヒストグラムのパラメータを変更する
df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
P.show()