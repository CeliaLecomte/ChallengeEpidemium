from sklearn import tree
from sklearn import ensemble
import pandas as pd
from sklearn.model_selection import train_test_split
from categorical_encoder import categorical_encoding
import matplotlib.pyplot as plt

def dectree(df,variable):

    df = df.loc[df['type'] == "C16"]
    df = categorical_encoding(df)
    print(len(df.columns.values))
    Y = df[variable]
    Y = (Y - Y.min())/(Y.max()-Y.min())
    # X = df.drop(variable, axis=1)
    X = df.drop(columns = [variable,'year'])
    print(X.columns.values)
    print(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=5)

    tr = tree.DecisionTreeRegressor()
    # tr = ensemble.RandomForestRegressor()
    print("fit")
    tr.fit(X_train,Y_train)
    print(tr.feature_importances_)
    print("predict")
    # y = tr.predict(X_test)
    # print(y)
    score = tr.score(X_test, Y_test)
    for i in range(len(tr.feature_importances_)):
        print("{}".format(str(list(X.columns.values)[i])))
        print("{}".format(str(tr.feature_importances_[i])))

    print(score)

df = pd.read_csv("C:/Users/titou/Desktop/Centrale/Option OSY/Projet/Datasets/ALL_PCA2_50.csv")
print(df.columns.names)
dectree(df, "relative_mortality")
