##on garde que certaines colonnes de FAO : celles où il y a Feed dedans


import pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




df=pd.read_csv('Fao_renamed.csv')
c=df.columns
n,l=df.shape


for j in range(3,l):
    print(j)
    if not('Feed' in c[j]):
        del(df[c[j]])

n,l=df.shape
df.to_csv('Fao_renamed_feed_only.csv')

