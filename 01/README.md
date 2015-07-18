# 01 titanic
タイタニック号の乗客データをもとに、乗客の生存／死亡を判定する問題  
<https://www.kaggle.com/c/titanic>  

訓練データとテストデータのダウンロード  
<https://www.kaggle.com/c/titanic/data>


###この問題で正解率はどれくらい出せるか？
統計モデルを使った予測としては0.8以上は良スコアだが、0.9～1.0はチートでないと
出ないスコアとのこと。「女性のみ全員生存」という単純な予測をしても0.76が出せるので、
かっこいい名前の統計モデルを使ったからと言って、飛躍的にスコアが
伸びるというわけでもない模様。

 <https://www.kaggle.com/c/titanic/forums/t/4894/what-accuracy-should-i-be-aiming-for>

参考スコア  
<https://www.kaggle.com/c/titanic/leaderboard>
- 全員死亡(assume all perished):0.626
- 女性のみ生存(gender based model):0.7655
- ランダムフォレスト(My first random forrest):0.77512
- 性別、クラス、運賃による予測(Gender, Price and Class Based Model):0.77990

###参考になるスクリプトなど  
####公開スクリプト一覧
<https://www.kaggle.com/c/titanic/scripts>  

- ランダムフォレスト with R  
kaggle公式<https://www.kaggle.com/c/titanic/details/new-getting-started-with-r>
<https://www.kaggle.com/benhamner/titanic/random-forest-benchmark-r>

- ランダムフォレスト with Python  
kaggle公式<https://www.kaggle.com/c/titanic/details/getting-started-with-random-forests>　　
<https://www.kaggle.com/thebrocean/titanic/benchmarking-random-forests>


###自分の提出済みデータと得点:  
<https://www.kaggle.com/c/titanic/submissions/>  


###ランダムフォレスト参考資料  
- 近畿大学  
<http://www.habe-lab.org/habe/RFtutorial/CVIM_RFtutorial.pdf>

- スライドシェア：機会学習ハッカソ:ランダムフォレスト  
<http://www.slideshare.net/teppeibaba5/ss-37143977>

- shakezoの日記
<http://d.hatena.ne.jp/shakezo/20121221/1356089207>

- スライドシェア：はじめてでもわかるランダムフォレスト
<http://www.slideshare.net/hamadakoichi/randomforest-web?ref=http://d.hatena.ne.jp/shakezo/20121221/1356089207>
