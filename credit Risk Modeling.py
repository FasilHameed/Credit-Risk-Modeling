# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import os
from warnings import filterwarnings

# Ignoring warnings
filterwarnings("ignore")


# Load Dataset
# Read the excel files
a1 = pd.read_excel('case_study1.xlsx')
a2 = pd.read_excel('case_study2.xlsx')

# Make Copies of the dataset
df1 = a1.copy()
df2 = a2.copy()

# Remove nulls from df1
# Drop rows where 'Age_Oldest_TL' is -99999
df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]

# Removing columns with more Null values from df2
columns_to_be_removed = []
# Identify columns with more than 10000 null values
for i in df2.columns:
    if df2.loc[df2[i] == -99999].shape[0] > 10000:
        columns_to_be_removed.append(i)

# Drop identified columns
df2 = df2.drop(columns_to_be_removed, axis=1)

# Removing nulls from all columns in df2
for i in df2.columns:
    df2 = df2.loc[df2[i] != -99999]

# combining datasets
df=pd.merge(df1,df2,on=['PROSPECTID'],how='inner')


