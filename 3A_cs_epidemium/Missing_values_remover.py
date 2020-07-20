import pandas as pd, numpy as np


# Recipe inputs
# data_WORLDBANK_df = pd.read_csv("C:/Users/titou/Desktop/Centrale/Option OSY/Projet/Datasets/WORLDBANK.csv")

def remove_na(df,ratio):
    col_names = list(df.columns.values)
    print(col_names)
    filtered_col_names = []
    for col_name in col_names:
        col = df[col_name]
        nb_nan = col.isnull().sum()
        if nb_nan/float(len(col)) < ratio :
            filtered_col_names += [col_name]
            print(len(filtered_col_names))
    output_df = df[filtered_col_names]
    return output_df
