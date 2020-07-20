
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
from tqdm import tqdm


df=pd.read_csv('BDD_decalage.csv')
df2=pd.read_csv('BDD_decalage_pour_prediction.csv')
caux=df.columns


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
c2=df2.columns

for i in range(n):
    tab[i,-1]=tab[i,-1]*100000


k=int(0.8*n)
        


xtrain=tab[:k,  3:l-1].copy()
ytrain=tab[:k,l-1].copy()
xtest=tab[k:n,3:l-1].copy()
ytest=tab[k:n,l-1].copy()
x_prediction=tab2[:,3:l-1].copy()



scaler = preprocessing.StandardScaler().fit(xtrain)
xtrain_scaled=scaler.transform(xtrain)
xtest_scaled=scaler.transform(xtest)
x_prediction_scaled=scaler.transform(x_prediction)


#clf=MLPRegressor()
clf = MLPRegressor(hidden_layer_sizes=(189,189), activation='tanh', solver='sgd', alpha=9, batch_size=124, learning_rate='adaptive', learning_rate_init=0.004, power_t=0.5, max_iter=10000, shuffle=False, random_state=False, tol=0.0001, verbose=True, warm_start=True, momentum=0.95, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.3, beta_1=0.99, beta_2=0.9999, epsilon=1e-09)

clf.fit(xtrain_scaled,ytrain)
prediction=clf.predict(xtest_scaled)
vrai_prediction=clf.predict(x_prediction_scaled)

score=mean_squared_error(ytest,prediction)
score2=r2_score(ytest,prediction)
score3=np.mean(abs(ytest-prediction)/ytest)
score4=max(abs(ytest-prediction)/ytest)
print(score,score2,score3,score4)



plt.close()
mini=min(ytest)
maxi=max(ytest)

plt.plot(ytest,prediction,marker='.', markersize=2, linewidth=0)
plt.plot([mini,maxi],[mini,maxi],linewidth=0.8)
plt.xlabel('taux multiplié par 100.000 réel')
plt.ylabel('taux multiplié par 100.000 prédit')
plt.show()



#ordre pour ajuster les paramètres en bouclant

resultat=np.concatenate((tab2[:,:2],np.reshape(vrai_prediction,(n2,1))),axis=1)
for i in range(n2):
    resultat[i,1]+=10
c=pd.Index(['Country','Year','Predicted mortality rate'])
df=pd.DataFrame(resultat,columns=c)

df.to_csv('previsionsNN_BDD_decalage.csv')



    

    
    
    
    
    
    
    
    
    
    
    