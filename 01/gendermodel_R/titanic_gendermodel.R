#UTF-8

#title:titanic gender変数を中心として
#author:si

#参照先http://trevorstephens.com/post/72916401642/titanic-getting-started-with-r
#ジェンダー変数を中心としたモデルルを学ぶ
train <- read.csv("../data/train.csv")

###タイタニック乗船者数を性別ごとに確認する
summary(train$Sex)

###性別ごとの生存率を見てみる
prop.table(table(train$Sex, train$Survived))

####ちなみに、tableに2変数以上を渡してやると、度数のクロス表を作ってくれる
table(train$Sex, train$Survived, train$Embarked, train$Pclass)
#*tableに渡す変数が2つより多くなると、はじめの2つの変数でクロス表が構成され、
#*そのあとに続く変数の要素ごとに（はじめの２変数で作られた）クロス表が生成される

##性別ごとの生存率を見てみる（行方向(row-wise)のproportionを1としている）
prop.table(table(train$Sex, train$Survived),1)
#*性別ごとの生存率と死亡率を表している
#*引数の1はrow-wiseを指定。2だとcolumn-wiseを指定することになる


##testファイルのSurvived情報を書き換える
test <- read.csv("../data/test.csv")
test$Survived <- 0
test$Survived[test$Sex == 'female'] <- 1
#*女性は生存率が高いから，女性は生き延びるってことにしておく

###次に年齢変数を検討する
###年齢の分布を確認
summary(train$Age)
#* 各種統計量の要約をsummaryでチェック
#* 年齢情報のないデータが177あるのがちょっと厄介
#* 連続型のデータはproportionを算出するのに役に立たない。
#* カテゴリー化した方が年齢別の生存率の傾向を割り出すのに便利（本当か？正確性を失わないの？）

###18才未満を子供とする変数を作る
train$Child <- 0
train$Child[train$Age < 18] <- 1 
#* この段階では，年齢のデータはすべてゼロになっている（のちに平均値を代入することになるらしい）
#* Rでは，変数を追加するtransform()という関数もあるらしい

###生存者のみについて，子供×性別でグループ分けをして集計
aggregate(Survived ~ Child + Sex, data=train, FUN=sum)
#* aggregate関数は，グループごとに集計する関数。この場合，
#* 子供とnon子供グループに分け，性別でグループにわけて，生存者数を集計している
#* aggregateには，aggregate(x,by,FUN)という書き方もあるらしいので，後で調べる
#* 引数のFUNで統計量を指定する

###乗船者全員について，子供×性別でグループ分けをして集計
aggregate(Survived ~ Child + Sex, data=train, FUN=length)
#* 生存者と死亡者，つまり，全乗船者を集計対象としている点に注意
#* FUNでlengthを指定することで全乗船者を対象としている。Survivedの0と1のどちらもカウント

###Subsetごとに生存率を算出する
###子供変数をみても，生死を分ける決定的な要因になっていない(っぽい)ことが分かる
#* aggregate(Survived ~ Child + Sex, data=train, function(x){(sum(x)/length(x))})
#* We need to create a function that takes the subset vector as input and applies both the sum and length commands to it, and then does the division to give us a proportion
#* function(x)に続く{}内の処理を各Subsetに対して実行する(?)

###船賃に注目
###船賃を連続データからカテゴリーデータに変える
train$Fare2 <- '30+'
train$Fare2[train$Fare < 30 & train$Fare >= '20'] <- '20-30'
train$Fare2[train$Fare < 20 & train$Fare >= '10'] <- '10-20'
train$Fare2[train$Fare <10] <- '<10'
#* 4つのカ???ゴリー???ータに変換した

###Subsetの変数にFare2を加えて、生存率を算出する
###Pclassは検証していないが，levelも多くないし，とりあえず変数に加えてみるかという程度に登場
aggregate(Survived ~ Fare2 + Pclass + Sex, data= train, function(x){sum(x)/length(x)})
# ここは変数を色々と変えてみると面白い。水準の低い変数は右辺のより右の項に置いたほうが表が見やすい

#* ここでわかったことは次の4つ
#* ・女性は生存率が高い
#* ・船賃で20＄以上を支払った女性の生存率は低い
#* ・クラス3の女性の生存率は低い
#* ・男性でPlass1で，20$以上支払った人の生存率はちょっと高い(tutolialには書いていない)

###上記のファインディングスに従って、testデータを書き換える
test$Survived <- 0
test$Survived[test$Sex == "female"] <- 1
test$Survived[test$Sex == "female" & test$Pclass == 3 & test$Fare >= 20] <- 0
test$Survived[test$Sex == "male"] <- 0
test$Survived[test$Sex == "male" & test$Pclass == 1 & test$Fare >= 20] <- 1


###Kaggleに提出するcsvファイルを作成する
submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit, "titanic_gendermodel.csv", row.names=FALSE)

