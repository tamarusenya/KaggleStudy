##データの確認
trainデータ  

    train_df = pd.read_csv('../../data/train.csv',header=0)
    print train_df.info()

    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object

testデータ  

    test_df = pd.read_csv('../../data/train.csv',header=0)
    test_df.info()

    Data columns (total 11 columns):
    PassengerId    418 non-null int64
    Pclass         418 non-null int64
    Name           418 non-null object
    Sex            418 non-null object
    Age            332 non-null float64
    SibSp          418 non-null int64
    Parch          418 non-null int64
    Ticket         418 non-null object
    Fare           417 non-null float64
    Cabin          91 non-null object
    Embarked       418 non-null object

##クレンジングの方針  
- Name,Ticket,Cabinについては、客の生死と関わりがあるか不明なので、無視
- object型のSex,Embarkedは、int型に変換する(Gender,Port)
- Age,Fareの欠損は、中央値で代替する(AgeFill,FareFill)
- Embarkedの欠損は、最頻値で代替する(PortFill)

trainデータのクレンジング後  

    train_mng_df = pd.read_csv('train_mng.csv',index_col=0, header=0)
    train_mng_df.info()

    Data columns (total 8 columns):
    Survived    891 non-null int64
    Pclass      891 non-null int64
    SibSp       891 non-null int64
    Parch       891 non-null int64
    Gender      891 non-null int64
    AgeFill     891 non-null float64
    FareFill    891 non-null float64
    PortFill    891 non-null float64

testデータのクレンジング後  

    test_mng_df = pd.read_csv('test_mng.csv',index_col=0, header=0)
    test_mng_df.info()

    Data columns (total 7 columns):
    Pclass      418 non-null int64
    SibSp       418 non-null int64
    Parch       418 non-null int64
    Gender      418 non-null int64
    AgeFill     418 non-null float64
    FareFill    418 non-null float64
    PortFill    418 non-null float64
