from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from categorical_encoder import categorical_encoding

def lasso_reg(df,variable):

    # df = df.loc[df['area'] != "Kuwait"]
    df = df.loc[df['type'] == "C16"]
    df = categorical_encoding(df)


    Y = df[variable]
    Y = (Y - Y.min()) / (Y.max() - Y.min())
    X = df.drop(variable, axis = 1)

    #Compute training amd testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state = 5)

    n_alphas = 10
    alphas = np.logspace(-5, 1, n_alphas)

    lasso = linear_model.Lasso(fit_intercept=False) #Data already centered
    # lasso = linear_model.LinearRegression()
    # lasso.fit(X_train,Y_train)

    coefs = []
    errors = []
    for a in alphas:
        lasso.set_params(alpha=a)
        lasso.fit(X_train, Y_train)
        coefs.append(lasso.coef_)
        errors.append(np.mean((lasso.predict(X_test) - Y_test) ** 2))
        errors.append(lasso.score(X_test,Y_test))

    #Plotting error
    ax = plt.gca()

    ax.plot(alphas, errors)
    ax.set_xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('error')
    plt.axis('tight')
    plt.show()

    #Select alpha that minimizes the error
    alpha = errors.index(min(errors))
    print("Le alpha minimisant l'erreur est : {}".format(str(alpha)))

    lasso.set_params(alpha= alpha)
    lasso.fit(X_train,Y_train)

    score = lasso.score(X_test,Y_test)
    print("Le score sur le testing set est : {}".format(str(score)))


df = pd.read_csv("C:/Users/titou/Desktop/Centrale/Option OSY/Projet/Datasets/ALL_PCA2_50_lag15.csv")
lasso_reg(df, "sum")


