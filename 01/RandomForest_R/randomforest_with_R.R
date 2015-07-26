# This script trains a Random Forest model based on the data,
# saves a sample submission, and plots the relative importance
# of the variables in making predictions

# Download 1_random_forest_r_submission.csv from the output below
# and submit it through https://www.kaggle.com/c/titanic-gettingStarted/submissions/attach
# to enter this getting started competition!

#libraryを読み込む
library(ggplot2)
library(randomForest)

#libraryを読み込む
#set.seedは乱数を記憶しておく装置
set.seed(1)

#train dataとtest dataを読み込む
train <- read.csv("../train.csv", stringsAsFactors=FALSE)
test  <- read.csv("../test.csv",  stringsAsFactors=FALSE)

#extractFeatures()という関数を定義する
#各変数をfreaturesでまとめる
extractFeatures <- function(data) {
  features <- c("Pclass",
                "Age",
                "Sex",
                "Parch",
                "SibSp",
                "Fare",
                "Embarked")
  fea <- data[,features]
  
  # NAに-1を代入
  fea$Age[is.na(fea$Age)] <- -1 
  # NAに運賃の中央値を代入
  fea$Fare[is.na(fea$Fare)] <- median(fea$Fare, na.rm=TRUE)
  # ==""で文字列かどうかを判定。TrueならＳを代入
  fea$Embarked[fea$Embarked==""] = "S"
  # ベクトルの要素を仲間同士を近くに並べて新しいベクトルを作る
  fea$Sex      <- as.factor(fea$Sex)
  fea$Embarked <- as.factor(fea$Embarked)
  return(fea)
}
#参考(fea$Age[is.na(fea$Age)])#######
# P <- matrix(c(1,2,NA,4),2,2)
# P[is.na(P)] <- 1
# P
#####################################

#上で作った関数extractFeaturesにtrainデータを代入
#ntreeは木の数。デフォルトは500
rf <- randomForest(extractFeatures(train), as.factor(train$Survived), ntree=100, importance=TRUE)

# testのIDを使った新しいデータフレームを作る
submission <- data.frame(PassengerId = test$PassengerId)
#predictは，モデル（ここではrf）と，データフレームを与えると応答変数を返してくれる
submission$Survived <- predict(rf, extractFeatures(test))
write.csv(submission, file = "1_random_forest_r_submission.csv", row.names=FALSE)

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

ggsave("2_feature_importance.png", p)
