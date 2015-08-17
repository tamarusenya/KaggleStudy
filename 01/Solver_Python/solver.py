# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pylab as P
import csv
import matplotlib.pyplot as plt
from sklearn import svm, cross_validation, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier

"""
１．データクリーニング
 １．trainデータと、testデータを統合する(コードが二度手間になるので)
 ２．sex,embarked,Pclass,Cabinをカテゴリデータとして2値化
 ３．Nameのうち生存率の高い上位10名の名前を、生存率で得点化
 4．age,fareの欠損値を、中央値で埋める
 5．最大値が1以上となるデータは、最大値1として正規化
 6.不要な列の削除(任意)
２．予測
 1.線形SVC
 2.非線形SVC
 3.ロジスティック回帰
 4.KNN近傍法
 5.ランダムフォレスト
"""

#CrossValidationの分割数
cv = 10

#データの読み込み
train_df = pd.read_csv('../data/train.csv',header=0)
test_df = pd.read_csv('../data/test.csv',header=0)

#テストデータにないSurvived列を追加し、-1を入力
test_df['Survived'] = -1

#１．trainデータと、testデータを統合する(コードが二度手間になるので)
df = pd.concat([train_df,test_df])
#欠損値に-1を入れる
df = df.fillna(-1)

#２．sex,embarked,Pclass,Cabinをカテゴリデータとして2値化
df['Gender'] = df['Sex'].map( {'female': 0, 'male':1} ).astype(int)
df['PortC'] = df['Embarked'].map( {'C': 1, 'Q':0, 'S':0, -1:0} ).astype(int)
df['PortQ'] = df['Embarked'].map( {'C': 0, 'Q':1, 'S':0, -1:0} ).astype(int)
df['PortS'] = df['Embarked'].map( {'C': 0, 'Q':0, 'S':1, -1:0} ).astype(int)
df['Pclass1'] = df['Pclass'].map( {1: 1, 2:0, 3:0, -1:0} ).astype(int)
df['Pclass2'] = df['Pclass'].map( {1: 0, 2:1, 3:0, -1:0} ).astype(int)
df['Pclass3'] = df['Pclass'].map( {1: 0, 2:0, 3:1, -1:0} ).astype(int)

#"""
#Cabin：一文字目のアルファベットで分類
#(客室は、アルファベットが早いものほど、船体上部に配置されている)
df['CabinClass'] = df['Cabin'].str[0:1]
df['CabinClass'] = df['CabinClass'].fillna(-1)
df['CabinClassA'] = df['CabinClass'].map( {'A':1, 'B':0,'C':0,'D':0,'E':0,'F':0,'G':0,'T':0, -1:0}).astype(int)
df['CabinClassB'] = df['CabinClass'].map( {'A':0, 'B':1,'C':0,'D':0,'E':0,'F':0,'G':0,'T':0, -1:0}).astype(int)
df['CabinClassC'] = df['CabinClass'].map( {'A':0, 'B':0,'C':1,'D':0,'E':0,'F':0,'G':0,'T':0, -1:0}).astype(int)
df['CabinClassD'] = df['CabinClass'].map( {'A':0, 'B':0,'C':0,'D':1,'E':0,'F':0,'G':0,'T':0, -1:0}).astype(int)
df['CabinClassE'] = df['CabinClass'].map( {'A':0, 'B':0,'C':0,'D':0,'E':1,'F':0,'G':0,'T':0, -1:0}).astype(int)
df['CabinClassF'] = df['CabinClass'].map( {'A':0, 'B':0,'C':0,'D':0,'E':0,'F':1,'G':0,'T':0, -1:0}).astype(int)
df['CabinClassG'] = df['CabinClass'].map( {'A':0, 'B':0,'C':0,'D':0,'E':0,'F':0,'G':1,'T':0, -1:0}).astype(int)
df['CabinClassT'] = df['CabinClass'].map( {'A':0, 'B':0,'C':0,'D':0,'E':0,'F':0,'G':0,'T':1, -1:0}).astype(int)
df = df.drop(['CabinClass'],axis=1)
#df['CabinClass'] = df['CabinClass'].map( {'A':0, 'B':0,'C':0,'D':0,'E':1,'F':1,'G':1,T':1,-1:-1}).astype(int)
#"""

#３．Nameのうち生存率の高い上位10名の名前を、生存率で得点化
#"""
df['NameLuck'] = -1
df['NameLuck'][train_df['Name'].str.contains('Elizabeth') == True] = 1.0
df['NameLuck'][train_df['Name'].str.contains('Anna') == True] = 0.8
df['NameLuck'][train_df['Name'].str.contains('Mary') == True] = 0.7
df['NameLuck'][train_df['Name'].str.contains('George') == True] = 0.3
df['NameLuck'][train_df['Name'].str.contains('Charles') == True] = 0.25
df['NameLuck'][train_df['Name'].str.contains('William') == True] = 0.25
df['NameLuck'][train_df['Name'].str.contains('Henry') == True] = 0.18
df['NameLuck'][train_df['Name'].str.contains('Thomas') == True] = 0.18
df['NameLuck'][train_df['Name'].str.contains('john') == True] = 0.15
#"""

#4．age,fareの欠損値を、中央値で埋める
#年齢の中央値を計算(Pclass,Gender別)
df_median_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        df_median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1) & (df['Age'] != -1)]['Age'].median()
#年齢の欠損値に中央値を入れる
for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age == -1) & (df.Gender == i) & (df.Pclass == j+1), 'Age'] = df_median_ages[i,j]
#運賃の欠損値に中央値を入れる
df['Fare'][df['Fare'] == -1] = df['Fare'][df['Fare'] != -1].median()

"""
#df['CabinNo'] = df['Cabin'].str[1:3]
#df['CabinNo'] = df['CabinNo'].fillna(-1)
#df['CabinNo'].to_csv('CabinNo.csv')
"""

#print df.info()

#'''
#5．最大値が1以上となるデータは、最大値1として正規化
df['Age'] = df['Age']/df['Age'].max()
df['Fare'] = df['Fare']/df['Fare'].max()
df['SibSp'] = df['SibSp']/df['SibSp'].max()
df['Parch'] = df['Parch']/df['Parch'].max()
#'''

#あとでデータフレームのカラムをアルファベット順に並び変える際、
#Survivedが、0列目になるよう、名前を変更
df['0_Survived'] = df['Survived']
df = df.drop(['Survived'],axis=1)
#関係ないデータを削除
df = df.drop(['Pclass','Name','Ticket','Cabin','Sex','Embarked','PassengerId'],axis=1)
#列をソート(0_Survivedが0列目になる)
df = df.sort(axis=1,)

#予測に使用しないデータを決定
df = df.drop([#'Age',
#              'Fare',
#              'Gender',
#              'Parch',
#              'Pclass1',
#              'Pclass2',
#              'Pclass3',
#              'PortC',
#              'PortQ',
#              'PortS',
#              'SibSp'
              ],axis=1)


#調整後のデータ
cln_df = df

#"""
#調整後のデータ情報
print cln_df.info()
#全データの最大値と最小値
#print '--------------'
#print cln_df.max(),cln_df.min()
#print '--------------'
#'''

#CSV出力
cln_df.to_csv('output/cln_data.csv',index=False)
#scikitlearnで解析るつため、numpyに変換
cln_data = cln_df.values


"""-------------------------------------------
線形サポートベクター分類
"""
#"""
#モデル作成
LinSVM_test = svm.LinearSVC(penalty = 'l1',
                            loss='hinge',
                            multi_class='crammer_singer',
                            dual=False,
                            max_iter=2000)
LinSVM_test.fit(cln_data[0:890,1::],cln_data[0:890,0])

#交叉検定によるモデル評価
CV_LinSVM = cross_validation.cross_val_score(LinSVM_test,cln_data[0:890,1::],cln_data[0:890,0],cv=cv)
CV_LinSVM = (sum(CV_LinSVM)/cv)
print 'CV(LinSVM)'
print CV_LinSVM
print '---------------'

#モデルによる予測と出力
output_LinSVM = LinSVM_test.predict(cln_data[891::,1::])
df_LinSVM = pd.DataFrame(output_LinSVM,columns = ['Survived'])
df_LinSVM.to_csv('output/LinSVM.csv',index=False)
#"""


"""-------------------------------------------
非線形サポートベクター分類
"""
"""
#モデル作成
NuSVC_test = svm.NuSVC()
NuSVC_test.fit(cln_data[0:891,1::],cln_data[0:891,0])

#交叉検定によるモデル評価
CV_NuSVC = cross_validation.cross_val_score(NuSVC_test,cln_data[0:891,1::],cln_data[0:891,0],cv=cv)
CV_NuSVC = print(sum(CV_NuSVC)/cv)
print 'CV(NuSVC)'
print CV_NuSVC
print '---------------'

#モデルによる予測と出力
output_NuSVC = NuSVC_test.predict(cln_data[892::,1::])
df_NuSVC = pd.DataFrame(output_NuSVC,columns = ['Survived'])
df_NuSVC.to_csv('output/NuSVC.csv',index=False)
"""


"""-------------------------------------------
ロジスティック回帰
"""
#"""
#モデル作成
logit_test = LogisticRegression(penalty = 'l1', #default
                                dual = False, #default
                                tol = 0.0001, #default
                                C=1.0, #default
                                fit_intercept = True, #default
                                intercept_scaling = 1, #default
                                class_weight = None, #default
                                random_state = None, #default
                                solver = 'liblinear', #default
                                max_iter = 100, #default
                                multi_class = 'ovr', #default
                                verbose = 0
                                )
# solver : {‘newton-cg’, ‘lbfgs’, ‘liblinear’}                               
                                

logit_test = logit_test.fit(cln_data[0:890,1::],cln_data[0:890,0])

#交叉検定によるモデル評価
CV_LogitRegrssion = cross_validation.cross_val_score(logit_test,cln_data[0:890,1::],cln_data[0:890,0],cv=cv)
CV_LogitRegrssion = (sum(CV_LogitRegrssion)/cv)
print 'CV(LogitRegrssion)'
print CV_LogitRegrssion
print '---------------'

#モデルの予測と出力
output_Logit = logit_test.predict(cln_data[891::,1::])
df_Logit = pd.DataFrame(data = output_Logit, columns = ['Survived'])
df_Logit.to_csv('output/Logit.csv',index=False)
#"""

"""-------------------------------------------
k近傍法
"""
#"""
#モデル生成
knn_test = KNeighborsClassifier(6)
knn_test = knn_test.fit(cln_data[0:890,1::],cln_data[0:890,0])

#交叉検定によるモデルの評価
CV_knn = cross_validation.cross_val_score(knn_test,cln_data[0:890,1::],cln_data[0:890,0],cv=cv)
CV_knn = (sum(CV_knn)/cv)
print 'CV(knn)'
print CV_knn
print '------'

#モデルの予測と出力
output_knn = knn_test.predict(cln_data[891::,1::])
df_knn = pd.DataFrame(output_knn,columns = ['Survived'])
df_knn.to_csv('output/knn.csv',index=False)
#"""

"""-------------------------------------------
ランダムフォレスト
"""
#"""
#モデルの生成
forest_test = RandomForestClassifier(n_estimators = 100,
                                     #criterion = "entropy" ,
                                     max_features = None)
forest_test = forest_test.fit(cln_data[0:890,1::],cln_data[0:890,0])

#交叉検定によるモデルの評価
CV_RandomForest = cross_validation.cross_val_score(forest_test,cln_data[0:890,1::],cln_data[0:890,0],cv=cv)
CV_RandomForest = (sum(CV_RandomForest)/cv)
print 'CV(RandomForest)'
print CV_RandomForest
print '---------------'

#モデルの予測と出力
output_RF = forest_test.predict(cln_data[891::,1::])
df_RF = pd.DataFrame(output_RF,columns = ["Survived"])
df_RF.to_csv('output/RForest.csv',index=False)
#"""


#"""
CV_all = np.array([CV_LinSVM,
                   CV_LogitRegrssion,
                   CV_knn,
                   CV_RandomForest])
ave = np.average(CV_all)
print(u"AVERAGE："+str(ave))
#"""