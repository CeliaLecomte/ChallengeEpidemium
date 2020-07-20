
from sklearn import preprocessing

import pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd
import math as ma 

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


df=pd.read_csv('BDD_decalage.csv')
df2=pd.read_csv('BDD_decalage_pour_prediction.csv')



#df=df.sort_values(by = 'Year')

del(df['Unnamed: 0'])
del(df['Unnamed: 0.1'])
del(df2['Unnamed: 0'])
del(df2['Unnamed: 0.1'])

c=df.columns

n,l=df.shape
n2,l2=df2.shape
tab=df.as_matrix()
tab2=df2.as_matrix()

k=int(0.8*n)

xtrain=tab[:k,  3:l-1].copy()
ytrain=tab[:k,l-1].copy()
xtest=tab[k:n,3:l-1].copy()
ytest=tab[k:n,l-1].copy()
x_prediction=tab2[:,3:l-1].copy()

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

regr = RandomForestRegressor(n_estimators=20, criterion='mae', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=True, n_jobs=1, random_state=None, verbose=100, warm_start=True)

regr.fit(xtrain,ytrain)

prediction=regr.predict(xtest)
mini=min(ytest)
maxi=max(ytest)

plt.plot(ytest,prediction,marker='.', markersize=2, linewidth=0)
plt.plot([mini,maxi],[mini,maxi],linewidth=0.8)
plt.xlabel('taux multiplié par 100.000 réel')
plt.ylabel('taux multiplié par 100.000 prédit')
plt.show()



plt.legend()
plt.show()


score=mean_absolute_error(ytest,prediction)
score2=r2_score(ytest,prediction)
score3=np.mean(abs(ytest-prediction)/ytest)
print(score,score2,score3)
vrai_prediction=regr.predict(x_prediction)

resultat=np.concatenate((tab2[:,:2],np.reshape(vrai_prediction,(n2,1))),axis=1)
for i in range(n2):
    resultat[i,1]+=10
c=pd.Index(['Country','Year','Predicted mortality rate'])
df=pd.DataFrame(resultat,columns=c)

df.to_csv('previsionsRF_BDD_decalage.csv')
