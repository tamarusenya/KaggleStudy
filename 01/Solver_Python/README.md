# solver.py
データクリーニングとscikit-learnによる予測をする.

##データクリーニング
- 1.trainデータと、testデータを統合する(コードが二度手間になるので)
- 2.sex,embarked,Pclass,Cabinをカテゴリデータとして2値化
- 3.Nameのうち生存率の高い上位10名の名前を、生存率で得点化
- 4.age,fareの欠損値を、中央値で埋める
- 5.最大値が1以上となるデータは、最大値1として正規化
- 6.不要な列の削除(任意)  

##予測
以下の各手法で予測と交叉検定を行う.
- 1.線形SVC  
http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html  
- 2.非線形SVC  
http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html  
- 3.ロジスティック回帰  
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html  
- 4.KNN近傍法  
http://scikit-learn.org/stable/modules/neighbors.html  
- 5.ランダムフォレスト  
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html  
- 1~6の交叉検定の平均値
