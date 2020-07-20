import pandas as pd, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# df = pd.read_csv()

def linReg(df,var):
    #variable to explain
    Y = df[var]
    #all the other variables
    X = df.drop[var]

    #linear model
    lm = LinearRegression()
    lm.fit(X, Y)
    result = pd.DataFrame(zip(X.columns, lm.coef_), columns=['Features', 'Coefficients'])

    return result