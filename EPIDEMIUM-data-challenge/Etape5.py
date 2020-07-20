## on supprime des colonnes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


df=pd.read_csv('BDD_premier_gros_nettoyage.csv')
del(df['Unnamed: 0'])
#del(df['proportion_de_-45'])
#del(df['proportion_de_femme'])
del(df['Unnamed: 0.1'])

tab=np.array(df)
(n,l)=df.shape
c=df.columns
print('n,l',n,l)

    
def peut_pas_etre_negatif():
    positif=[]
    for j in range(2,l):
        i=0
        while(i<n and not(tab[i,j]<0)):
            i+=1
        if(i==n):
            positif.append(j)
    return(positif)
col_positif=peut_pas_etre_negatif()



liste_rouge_pays={}
for i in range(n):
    pays=tab[i,0]
    liste_rouge_pays[pays]=0
liste_rouge_colonnes=[0 for k in range(l) ]


for j in tqdm(range(2,l)):
    
    i=0
    deja_traite=[]
    while(i<n):
        
        pays=tab[i,0]
        
        if(not(pays in deja_traite)):
            
            x=[]
            y=[]
            k=i
            compteur=0
            nan=[]
            
            while(k<n):
                if(tab[k,0]==pays ):
                    
                    if(np.isfinite(tab[k,j])):
                        x.append(tab[k,1])
                        y.append(tab[k,j])
                        compteur+=1
                    else:
                        nan.append(k)
                k+=1
            if(compteur==0):
                liste_rouge_pays[pays]+=1
                liste_rouge_colonnes[j]+=1
         
            deja_traite.append(pays)
        i+=1
        
        
for j in range(2,l):
    
    valeur=0
    compteur=0
    nan=[]
    for i in range(n):
        if(not(np.isfinite(tab[i,j]))):
            nan.append(i)
        else:
            compteur+=1
            valeur+=tab[i,j]
           
        
        

print(n,l)
df=pd.DataFrame(tab,columns=c) 
for i in range(l):
    if(liste_rouge_colonnes[i]>3): #and not ('Meat' in c[i])):
        del(df[c[i]])




df.to_csv('BDD_deuxieme_nettoyage_colonnes.csv')
                
        
        
        
        
        
        
        
        
        