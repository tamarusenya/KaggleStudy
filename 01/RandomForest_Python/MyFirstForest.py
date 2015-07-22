# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pylab as P
import csv
from sklearn.ensemble import RandomForestClassifier 

#データの読み込み
train_df = pd.read_csv('../data/train.csv',header=0)
test_df = pd.read_csv('../data/test.csv',header=0)

#2.pandasを使い、欠損値を年齢の代表値で埋める
#年null値を確認
#print train_df[train_df['Age'].isnull()]
#print train_df[train_df['Fare'].isnull()]
#print test_df[test_df['Age'].isnull()]
#print test_df[test_df['Fare'].isnull()]

##female→0 male→1の整数に変換したものをGenderとして新しく定義する
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male':1} ).astype(int)
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male':1} ).astype(int)

##すべての要素が0の2行3列の"median_ages"行列をつくる
train_median_ages = np.zeros((2,3))
test_median_ages = np.zeros((2,3))

##上で作製したmedian_agesの各要素に、性別、クラス別の年齢中央値を入力する
for i in range(0, 2):
    for j in range(0, 3):
        train_median_ages[i,j] = train_df[(train_df['Gender'] == i) & (train_df['Pclass'] == j+1)]['Age'].dropna().median()
        test_median_ages[i,j] = test_df[(test_df['Gender'] == i) & (test_df['Pclass'] == j+1)]['Age'].dropna().median()

##AgeFillを定義する（今のところAgeと同じ）
train_df['AgeFill'] = train_df['Age']
test_df['AgeFill'] = test_df['Age']

##AgeFillの欠損箇所に、median_agesを入力する
##pandasのloc関数（locで条件付けられている要素の、AgeFillのみを抽出し、そこに中央値（i,j）を使う
##http://oceanmarine.sakura.ne.jp/sphinx/group/group_pandas.html#id75
for i in range(0, 2):
    for j in range(0, 3):
        train_df.loc[ (train_df.Age.isnull()) & (train_df.Gender == i) & (train_df.Pclass == j+1), 'AgeFill'] = train_median_ages[i,j]
        test_df.loc[ (test_df.Age.isnull()) & (test_df.Gender == i) & (test_df.Pclass == j+1), 'AgeFill'] = test_median_ages[i,j]

##Fareのnull値にFareの中央値を入れる
##(欠損値が一つだけなので,条件付きの中央値は使わない)

"""
test_df[test_df['Fare'].isnull() == True]['Fare'] = test_df['Fare'].dropna().median()
print test_df[test_df['Fare'].isnull() == True]['Fare']
"""

#3.pandasを使い、元データから特徴量を新しく作る
##兄弟と両親をファミリーサイズに統合
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch']
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']
train_df['Age*Class'] = train_df.AgeFill * train_df.Pclass        
test_df['Age*Class'] = test_df.AgeFill * test_df.Pclass        


##sklearn packageではpandasを扱えないので、最後にnumpyarrayに戻す



##データ内の、データ型がobjectのもののみを表示する
#print train_df.dtypes[train_df.dtypes.map(lambda x: x=='object')]


##test_dfのPassengerIdを後で使うため記憶しておく
ids = test_df['PassengerId'].values

##以下のデータは捨てる
##
##Age,Sex,SibSp,Parch(別で代替しているので）
##PassengerId,Name,Ticket,Cabin(推定に不要なため)
##Embarked,Fare(欠損値があるため。→今後導入の予定)

train_df = train_df.drop(['Age','Sex','SibSp','Parch','PassengerId','Name','Ticket','Cabin','Embarked','Fare'],axis=1)
test_df = test_df.drop(['Age','Sex','SibSp','Parch','PassengerId','Name','Ticket','Cabin','Embarked','Fare'],axis=1)

#データクリーン後のデータもCSVファイルにアウトプットしておく
#train_df.to_csv('train_df.csv')
#test_df.to_csv('test_df.csv')

train_data = train_df.values
test_data = test_df.values

#5.randomforestの適用
#https://www.kaggle.com/c/titanic/details/getting-started-with-random-forests

# Create the random forest object which will include all the parameters
# for the fit
##モデル(forest_test)の定義とパラメータの入力(n_estimators = 100)
forest_test = RandomForestClassifier(n_estimators = 100)
##モデルに訓練データtrain_data[0::,1::]と、訓練ラベルtrain_data[0::,0](Survived)を入れる
forest_test = forest_test.fit(train_data[0::,1::],train_data[0::,0])
##上で学習されたモデルにテストデータを入力し、予測結果をoutputとする
output = forest_test.predict(test_data)

##予測結果を入れるためのCSVファイルを作る
predictions_file = open('../data/MyFirstForest.csv', "wb")
#CSVファイルをpythonで開く
open_file_object = csv.writer(predictions_file)
#列名を書き込む
open_file_object.writerow(["PassengerId","Survived"])
#テストデータのPassengerIdと予測データを入れる
open_file_object.writerows(zip(ids, output))
#CSVファイルを閉じる
predictions_file.close()
