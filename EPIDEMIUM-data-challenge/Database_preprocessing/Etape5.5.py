## on remplit les trous et supprime des pays

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


df=pd.read_csv('BDD_deuxieme_nettoyage_colonnes_gardeviande.csv')
del(df['Unnamed: 0'])
#del(df['proportion_de_-45'])
#del(df['proportion_de_femme'])
#del(df['Unnamed: 0.1'])

tab=np.array(df)
(n,l)=df.shape
c=df.columns


    
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
            elif(compteur==1):
                for k in nan:
                    tab[k,j]=y[0]
            else:
                coefficients=np.polyfit(x,y,1)
                a=coefficients[0]
                b=coefficients[1]
                if(j in col_positif):
                    for k in nan:
                        tab[k,j]=max(0,a*tab[k,1]+b)
                else:
                    for k in nan:
                        tab[k,j]=a*tab[k,1]+b
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
    for i in nan:
        tab[i,j]=valeur/compteur        
        
        





def suppression_pays(nom,tab):
    n,l=tab.shape
    l2=[]
    compteur=0
    for i in range(n):
        if(tab[i,0]!=nom):
            l2.append(i)
            compteur+=1
    colonne=np.reshape(np.array(['A' for i in range(compteur)]),(compteur,1))
    tab2=np.concatenate((colonne,np.array([[1.1 for j in range(l-1)]for i in range(compteur)])),axis=1)
    #print(compteur)
    for i in range(compteur):
        tab2[i,0]=tab[l2[i],0]
        for j in range(1,l):
            tab2[i,j]=float(tab[l2[i],j])
    return(tab2)

  
#21 pour autre que viande
for pays in liste_rouge_pays.keys():
    if(liste_rouge_pays[pays]>31):
        tab=suppression_pays(pays,tab)
tab=suppression_pays('Belize',tab)
df=pd.DataFrame(tab,columns=c)

        


df.to_csv('BDD_troisieme_nettoyage_gardeviande.csv')