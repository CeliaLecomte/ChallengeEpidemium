##On supprime les colonnes qui ont plus de 95% manquant

import pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

            
df=pd.read_csv('BDD_rapporte_a_la_pop.csv')


tab=np.array(df)
(n,l)=df.shape
print(l)
c=df.columns
supprimer=[]
garder=[0,1,2]
for j in range(3,l):
    compteur=0
    for i in range(n):
        compteur+=(not (np.isfinite(tab[i,j])))   #on incrémente si on tombe sur une cellule vide
    if(compteur/n <0.95):
        garder.append(j)
    else:
        supprimer.append(j)
    
c2=[]
for k in range(len(supprimer)) :
    tab=np.delete(tab,(supprimer[k]-k),axis=1)
for j in garder:
    c2.append(c[j])
    
print(tab.shape)
        
        


#on supprime 34 colonnes








df=pd.DataFrame(tab,columns=c2) 
df.to_csv('BDD_premier_gros_nettoyage.csv')