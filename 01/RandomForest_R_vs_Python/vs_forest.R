# -*- coding: utf-8 -*-
# train_cln.csvとtest_cln.csvを使って、
#pythonとRのランダムフォレストのスコアを比較する
#tree:100
#情報量：エントロピー

#libraryを読み込む
library(ggplot2)
library(randomForest)

#libraryを読み込む
#set.seedは乱数を記憶しておく装置
set.seed(1)

#train dataとtest dataを読み込む
train <- read.csv("cleaning/train_cln.csv", stringsAsFactors=FALSE)
test  <- read.csv("cleaning/test_cln.csv",  stringsAsFactors=FALSE)

#extractFeatures()という関数を定義する
#各変数をfreaturesでまとめる
extractFeatures <- function(data) {
  features <- c("Pclass",
                "AgeFill",
                "Gender",
                "SibSp",
                "FareFill")
  fea <- data[,features]
  
  return(fea)
}


#上で作った関数extractFeaturesにtrainデータを代入
#ntreeは木の数。デフォルトは500
rf <- randomForest(extractFeatures(train), as.factor(train$Survived), ntree=100, importance=TRUE)

# testのIDを使った新しいデータフレームを作る
submission <- data.frame(PassengerId = test$PassengerId)
#predictは，モデル（ここではrf）と，データフレームを与えると応答変数を返してくれる
submission$Survived <- predict(rf, extractFeatures(test))
write.csv(submission, file = "vs_result/ForestR.csv", row.names=FALSE)

#importanceは，特徴量の重要度を計算する関数
#http://alfredplpl.hatenablog.com/entry/2013/12/24/225420
imp <- importance(rf, type=1)
featureImportance <- data.frame(Feature=row.names(imp), Importance=imp[,1])

p <- ggplot(featureImportance, aes(x=reorder(Feature, Importance), y=Importance)) +
  geom_bar(stat="identity", fill="#53cfff") +
  coord_flip() + 
  theme_light(base_size=20) +
  xlab("") +
  ylab("Importance") + 
  ggtitle("Random Forest Feature Importance\n") +
  theme(plot.title=element_text(size=18))

ggsave("vs_result/forestR_importance.png", p)
