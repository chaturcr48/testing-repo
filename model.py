import numpy as np
import pandas as pd

url = "https://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"

df = pd.read_csv(url)

df

include = ['Age', 'Sex', 'Embarked', 'Survived']
df_ = df[include]
# df_

categoricals=[]

# df_.dtypes

for col, col_type in df_.dtypes.iteritems():
    if col_type =='O':
        categoricals.append(col)
    else:
        df_[col].fillna(0, inplace=True)

# df_

df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

# df_ohe

from sklearn.linear_model import LogisticRegression
dependent_variable = 'Survived'
# df_ohe[df_ohe.columns.difference([dependent_variable])]
x = df_ohe[df_ohe.columns.difference([dependent_variable])]
# df_ohe[dependent_variable]
y = df_ohe[dependent_variable]

lr = LogisticRegression()
lr.fit(x,y)

import joblib
joblib.dump(lr, 'model.pkl')
print("Model dumped!")

lr = joblib.load('model.pkl')
model_columns = list(x.columns)
# model_columns
joblib.dump(model_columns, 'model_columns.pkl')
print("Model column dumped")