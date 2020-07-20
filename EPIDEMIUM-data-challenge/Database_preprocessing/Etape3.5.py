import pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

            
df=pd.read_csv('BDD_jointure_fini2.csv')


tab=np.array(df)
(n,l)=df.shape
c=df.columns

l2=[112, 143, 148, 149, 151, 168, 170, 175, 193, 194, 223, 226, 228, 231, 233, 235, 238, 242, 243, 244, 246, 250, 254, 266, 279, 282, 283, 288, 290, 292, 294, 295, 297, 299, 301, 303, 305, 307, 310, 311, 322, 324, 326, 327, 328, 333, 335, 346, 350, 361, 363, 365, 376, 380, 381, 382, 391, 491, 496, 497, 498, 499, 500, 501, 547]      

k=0
for j in range(l):
    if(c[j]=='pop'):
        k=j


for j in l2:
    for i in range(2):
        tab[i,j]=tab[i,j]/tab[i,k]

df=pd.DataFrame(tab,columns=c)
del(df['col_0'])
del(df['proportion_de_-45'])
del(df['proportion_de_femme'])
del(df['pop'])
df.to_csv('BDD_rapporte_a_la_pop.csv')