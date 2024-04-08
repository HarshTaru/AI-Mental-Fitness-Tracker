import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
df1 = pd.read_csv('assets/mental-and-substance-use-as-share-of-disease.csv')
df2 = pd.read_csv('assets/prevalence-by-mental-and-substance-use-disorder.csv')
df1.head()
df2.head()
data = pd.merge(df1, df2)
data.head()
data.isnull().sum()
data.drop('Code',axis=1,inplace=True)
data.head()
data.set_axis(['Country','Year','Schizophrenia', 'Bipolar_disorder', 'Eating_disorder','Anxiety','drug_usage','depression','alcohol','mental_fitness'], axis='columns')
data.head()

plt.figure(figsize=(12,6))
sns.heatmap(data.corr(),annot=True,cmap='Blues')
plt.plot()