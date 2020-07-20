##on decale le Dataset pour passer en prédiction dans le futur


import pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

            
df=pd.read_csv('BDD_troisieme_nettoyage.csv')
#del(df['Unnamed: 0'])
tab=np.array(df)
(n,l)=df.shape
c=df.columns
m=10
#renvoie tab avec les incidences décalés : features année n mortalité année n+m
def decalage(tab,n):
    tab2=np.concatenate((tab,np.reshape(np.zeros((n,1)),(n,1))),axis=1)
    i=0
    
    l2=[0 for k in range(n)]
    SUPPRIMER=[]
    while(i<n):
        pays=tab[i,1]
        année=tab[i,2]
        if(tab[i,2]==2008 and pays=='Azerbaijan'):
            print(i)
        j=0
        
        
        while(j<n):
            if (tab[j,1]==pays and tab[j,2]==année+m):
                l2[i]=tab[j,3]
                j=n
            j+=1
        if(j==n):
            SUPPRIMER.append(i)

        
        i+=1
    
    for k in range(n):
        
        tab2[k,-1]=l2[k]
    
    longueur=len(SUPPRIMER)
    colonne=np.reshape(np.array(['A' for i in range(longueur)]),(longueur,1))
    tab3=np.concatenate((colonne,np.array([[1.1 for j in range(l-1)]for i in range(longueur)])),axis=1)
    tab_prevision=np.concatenate((tab3,np.zeros((longueur,1))),axis=1)
    for k in range(longueur):
        for j in range(l):
            tab_prevision[k,j]=tab2[SUPPRIMER[k]-k,j]
        tab2=np.delete(tab2,(SUPPRIMER[k]-k),axis=0)
        
    return(tab_prevision,tab2)
    
tab_prevision,tab2=decalage(tab,n)

df2=pd.DataFrame(tab2,columns=pd.Index(list(c)+['mr année N+5']))
df=pd.DataFrame(tab_prevision,columns=pd.Index(list(c)+['mr année N+5']))

df.to_csv('BDD_decalage_pour_prediction.csv')
df2.to_csv('BDD_decalage.csv')





(n,l)=tab_prevision.shape
for i in range(n):
    if(tab_prevision[i,1]=='Azerbaijan' and  tab[i,2]==2007):
        print(2000,tab[i,3])
    elif(tab_prevision[i,1]=='Azerbaijan' and  tab[i,2]==2005):
        print(2005,tab[i,3])


            

            
            