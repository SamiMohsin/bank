import os

# to use amd gpu in mac
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# Don't use tensorflow.keras anywhere, instead use keras

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import matplotlib.pyplot as plt


import pandas as pd
import numpy as np

batch_size = 128
num_classes = 10
epochs = 12


colnames = [
'job',
'marital',
'education',
'default',
'balance',
'housing',
'loan',
'contact',
'day',
'month',
'duration',
'campaign',
'pdays',
'previous',
'poutcome',
'y'
]

df = pd.read_csv('data/bank.csv', sep=';')




target= df['y']
feats = df.drop('y', axis=1)


target.to_csv('data/target.csv', header='y')
feats.to_csv('data/feats.csv')

# default
default = df['default'].value_counts()
default.plot(kind='bar')
plt.show()
df['is_default'] = df['default'].apply(lambda x:1 if x == 'yes' else 0)
df['is_loan'] = df['loan'].apply(lambda x:1 if x == 'yes' else 0)


# marital
print(df['marital'].value_counts())
df['marital'].value_counts().plot(kind='bar')
plt.show()
marital_dummies = pd.get_dummies(df['marital'])
pd.concat([df['marital'], marital_dummies],  axis=1)
marital_dummies.drop('divorced', axis=1, inplace=True)
marital_dummies.columns = [f'marital_{colname}' for colname in marital_dummies.columns]
df = pd.concat([df, marital_dummies], axis=1)


# job
df['job'].value_counts().plot(kind='bar')
plt.show()
job_dummies = pd.get_dummies(df['job'])
job_dummies.drop('unknown', axis=1, inplace=True)
job_dummies.columns = [f'job_{colname}' for colname in job_dummies.columns]
df = pd.concat([df, job_dummies], axis=1)


# education
print(df['education'].value_counts())
df['education'].value_counts().plot(kind='bar')
plt.show()
education_dummies = pd.get_dummies(df['education'])
education_dummies.drop('unknown', axis=1, inplace=True)
education_dummies.columns = [f'education_{colname}' for colname in education_dummies.columns]
df = pd.concat([df, education_dummies], axis=1)



# month
month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may': 5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
print(df['month'].value_counts())
df['month'].value_counts().plot(kind='bar')
plt.show()
df['month'] = df['month'].map(month_map)



# poutcome
print(df['poutcome'].value_counts())
df['poutcome'].value_counts().plot(kind='bar')
plt.show()
poutcome_dummies = pd.get_dummies(df['poutcome'])
poutcome_dummies.drop('unknown', axis=1, inplace=True)
poutcome_dummies.columns = [f'poutcome_{colname}' for colname in poutcome_dummies.columns]
df = pd.concat([df, poutcome_dummies], axis=1)


# show columns
print(df.iloc[0])


df.drop(['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome'], axis=1, inplace=True)
print('_'*60)
print(df.dtypes)

"""
output:
age                     int64
balance                 int64
day                     int64
month                   int64
duration                int64
campaign                int64
pdays                   int64
previous                int64
y                      object
is_default              int64
is_loan                 int64
marital_married         uint8
marital_single          uint8
job_admin.              uint8
job_blue-collar         uint8
job_entrepreneur        uint8
job_housemaid           uint8
job_management          uint8
job_retired             uint8
job_self-employed       uint8
job_services            uint8
job_student             uint8
job_technician          uint8
job_unemployed          uint8
education_primary       uint8
education_secondary     uint8
education_tertiary      uint8
poutcome_failure        uint8
poutcome_other          uint8
poutcome_success        uint8
dtype: object
"""


feats = df



# pdays
feats['pdays'].hist(bins=50)
plt.show()
print(feats[feats['pdays']== -1]['pdays'].count())
feats[feats['pdays']> -1]['pdays'].hist(bins=50)
plt.show()


feats['was_contact'] = feats['pdays'].apply(lambda row: 1 if row>-1 else 0)
feats.drop('pdays', axis=1, inplace=True)



target = df['y'].apply(lambda x:1 if x == 'yes' else 0)
feats = df.drop('y', axis=1)


feats.to_csv('data/feats_e3.csv')
target.to_csv('data/target_e3.csv', header='y')





