from PCA import transform_pca
from categorical_encoder import categorical_encoding
from Missing_values_remover import remove_na
import pandas as pd


#Input
df = pd.read_csv("C:/Users/titou/Desktop/Centrale/Option OSY/Projet/Datasets/Worldbank_Replaced_Countries.csv", header =1)

names = list(df.columns.values)
print(names)

#Choose which column to keep
countries = df['area']
years = df['year']
pop = df['SP.POP.TOTL']
# rel_mort = df['relative_mortality']
# types = df['type']

for i in range(4):
    print(names[i])
    df = df.drop(names[i], axis= 1)


print(list(df.columns.values))
#Removing na values
df = remove_na(df,0.7)

#Fitting PCA
df = transform_pca(df)
names = list(df.columns.values)
print(names)
for i in range(len(names)):
    names[i] = "WB." + names[i]
df.columns = names

df.insert(0, 'SP.POP.TOTL', pop)
df.insert(0, 'year', years)
df.insert(0, 'area', countries)

df.to_csv("C:/Users/titou/Desktop/Centrale/Option OSY/Projet/Datasets/WB_PCA_30.csv", index= False)