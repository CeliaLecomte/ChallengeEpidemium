import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import datasets

# Load the Iris flower dataset:
iris = datasets.load_iris()
X = iris.data
y = iris.target
print(X)
print(y)

#
# df = pd.read_csv()
#
# def lda(df,var):
#     """Fucntion to compute LDA on a dataset. It chooses n_components so the variance explained is > var"""
#
#     sklearn_lda = LDA(n_components=None)
#
#     lda_var_ratios = sklearn_lda.explained_variance_ratio_
#
#     # Set initial variance explained so far
#     total_variance = 0.0
#
#     # Set initial number of features
#     n_components = 0
#
#     # For the explained variance of each feature:
#     for explained_variance in lda_var_ratios:
#
#         # Add the explained variance to the total
#         total_variance += explained_variance
#
#         # Add one to the number of components
#         n_components += 1
#
#         # If we reach our goal level of explained variance
#         if total_variance >= goal_var:
#             # End the loop
#             break
#
#     sklearn_lda_var = LDA(n_components=n_components)
#     df_lda = sklearn_lda_var.fit_transform()
