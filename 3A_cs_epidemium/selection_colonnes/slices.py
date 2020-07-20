import pandas as pd
import csv
import collections
import numpy as np

#Répartition des colonnes en fonction de leur % de valeurs manquantes pour un fichier csv donné
def columns_repartition(file_name):
    data_df = pd.read_csv(file_name)
    col_names = list(data_df.columns.values)
    print(col_names)
    number_valid_columns = np.zeros(20)
    percentages_list=[]
    for l in range(20):
        percentages_list+=[str(l*5)+'%']
    for i in range(20):
        percentage=((1/20)*i)
        for col_name in col_names:
            col = data_df[col_name]
            nb_nan = col.isnull().sum()
            if (nb_nan / len(col)) <= percentage :
                number_valid_columns[i]+=1
    print(number_valid_columns)
    print(percentages_list)
    with open(str(file_name)[:len(file_name)-4] + '_col_repart.csv', 'w') as file:
        for i in range(20):
            file.write(percentages_list[i] + ';' + str(number_valid_columns[i]) + '\n')

#Noms des colonnes à garder dans la base FAO pour un pourcentage maximal de valeurs manquantes donné
def keep_colums_fao(percentage):

    data_FAO_df = pd.read_csv('FAO.csv')
    col_names = list(data_FAO_df.columns.values)
    feature_list=[]
    for col_name in col_names:
            col = data_FAO_df[col_name]
            nb_nan = col.isnull().sum()
            if (nb_nan / len(col)) <= percentage :
                feature_list+=[col_name]

    print("Codes des colonnes à garder  : {}".format(feature_list))
    print("Nombre de colonnes à garder" + str(len(feature_list)))

    data_FAO_INDICATORS_df = pd.read_csv('FAO_INDICATORS.csv')
    indicateur_col = data_FAO_INDICATORS_df['Indicateur']
    item_col = data_FAO_INDICATORS_df['Item']
    element_col = data_FAO_INDICATORS_df['Element']
    filtered_indicators_list = []
    for i in range(len(feature_list)):
        for j in range(len(indicateur_col)):
            if feature_list[i]==indicateur_col[j]:
                filtered_indicators_list+=[item_col[j] + ";" + element_col[j]]
    print("Noms des colonnes conservées (hors pays et année) : {}".format(filtered_indicators_list))

    with open(str(percentage) + 'output_FAO.csv', 'w') as file:
        for i in range(len(filtered_indicators_list)):
            file.write(filtered_indicators_list[i] + '\n')

#Nom des colonnes à garder dans la base WB pour un pourcentage maximal de valeurs manquantes donné
def keep_colums_wb(percentage):

    data_WORLDBANK_df = pd.read_csv('WORLDBANK.csv')
    col_names = list(data_WORLDBANK_df.columns.values)
    feature_list=[]

    for col_name in col_names:
            col = data_WORLDBANK_df[col_name]
            nb_nan = col.isnull().sum()
            if (nb_nan / len(col)) <= percentage :
                feature_list+=[col_name]

    print("Codes des colonnes à garder  : {}".format(feature_list))
    print("Nombre de colonnes à garder" + str(len(feature_list)))

    data_WORLDBANK_INDICATORS_df = pd.read_csv('WORLDBANK_INDICATORS.csv')
    indicators_col_names = data_WORLDBANK_INDICATORS_df.columns.values
    code_col = data_WORLDBANK_INDICATORS_df['Code']
    name_col = data_WORLDBANK_INDICATORS_df['Indicator Name']
    filtered_indicators_list = []

    for i in range(len(feature_list)):
        for j in range(len(code_col)):
            if feature_list[i]==code_col[j]:
                filtered_indicators_list+=[name_col[j]]

    print("Noms des colonnes conservées (hors pays et année) : {}".format(filtered_indicators_list))

    with open(str(percentage) + 'output_WORLDBANK.csv', 'w') as file:
        for i in range(len(filtered_indicators_list)):
            file.write(filtered_indicators_list[i] + '\n')

