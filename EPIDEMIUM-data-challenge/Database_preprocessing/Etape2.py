##on join mortality, WB et FAO


import pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#df=pd.read_csv('Mortality_pepared.csv')
df=pd.read_csv('Mortality_modified.csv')
del(df['Unnamed: 0'])
c=df.columns

df2=pd.read_csv('Fao_renamed_feed_only.csv')
del(df2['Unnamed: 0'])
del(df2['Unnamed: 0.1'])
c2=df2.columns

df3=pd.read_csv('WorldBank_renamed.csv')
del(df3['area_code'])
c3=df3.columns

li=list(df2)
li[0]='Country'
li[1]='Year'
c2=pd.Index(li)
tab2=np.array(df2)
df2=pd.DataFrame(tab2,columns=c2)


li=list(df3)
li[0]='Country'
li[1]='Year'
c3=pd.Index(li)
tab3=np.array(df3)
df3=pd.DataFrame(tab3,columns=c3)



df=pd.merge(df, df2, on=['Country','Year'], how='left')
df=pd.merge(df, df3, on=['Country','Year'], how='left')

li=list(df.columns)
n,l=df.shape
tab=np.array(df)

for j in range(5,83):
    li[j]=li[j]+' (per capita)'
    for i in range(n):
        tab[i,j]=tab[i,j]/tab[i,488]

df=pd.DataFrame(tab,columns=pd.Index(li))
df.to_csv('BDD_jointure_fini.csv')




#on join fao et mortality puis on divise et ajout pourcent

#on join mortality et on amene a laure

#etape 3

#on remplit de 2 manières différentes

# on décale d'une manière
