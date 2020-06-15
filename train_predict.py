#Train_Predict.ipynb

import pandas as pd 
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn import preprocessing

loanf=pd.read_csv('loan_final.csv')

X=loanf.iloc[:,1:12].values
y=loanf.iloc[:,-1].values
loanf['Property_Area']=loanf['Property_Area'].astype(str)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [10])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 0)

pca=PCA()
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

classifier = XGBClassifier()
classifier.fit(X_train,y_train)

predict=classifier.predict(X_test)

cm = confusion_matrix(y_test,predict)
print(cm)

accuracy_score(y_test,predict)
