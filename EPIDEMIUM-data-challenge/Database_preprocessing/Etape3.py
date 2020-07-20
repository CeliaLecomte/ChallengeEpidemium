## On ramène les grandeurs liées à la taille de la population en grandeur par tête

import pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

            
df=pd.read_csv('BDD_jointure_fini2.csv') 
##2 pcq entre temps on a commencé à diviser par la population sur dataiku


tab=np.array(df)
(n,l)=df.shape
c=df.columns

l2=[]
for j in range(l):
    if(not('%' in c[j] or 'per' in c[j] or 'rate' in c[j] or 'index'in c[j])):
        print(df[c[j]])
        print(c[j])
        if(input()=='o'):
            l2.append(j)
        
            
