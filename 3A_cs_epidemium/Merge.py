import pandas as pd
import csv
import collections

# Recipe inputs

data_WORLDBANK_df = pd.read_csv('C:/Users/titou/Desktop/Centrale/Option OSY/Projet/Datasets/FAO_PCA_50.csv')
mortality_aggregate_df = pd.read_csv('C:/Users/titou/Desktop/Centrale/Option OSY/Projet/Datasets/WB_PCA_50.csv')

data_output = pd.merge(mortality_aggregate_df, data_WORLDBANK_df, how = 'inner', on = ['area', 'year'])
print(list(data_output.columns.values))
data_output = data_output.drop(columns = ['Unnamed: 0_x'])

data_output.to_csv('C:/Users/titou/Desktop/Centrale/Option OSY/Projet/Datasets/ALL_PCA_50.csv')