# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:55:58 2024

@author: danie
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import statsmodels.api as sm
import matplotlib.pyplot as plt


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

model = LogisticRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print (classification_report(Y_test, Y_pred))

confusion_mat = confusion_matrix(Y_test, Y_pred)

#feature importance 

coefficients = model.coef_[0]
feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})
feature_importance['Importance'] = np.abs(feature_importance['Coefficient'])

feature_importance = feature_importance.sort_values(by = 'Importance', ascending=False)
print(feature_importance)

plt.figure()
plt.bar(feature_importance['Feature'], feature_importance['Importance'], color = 'skyblue')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance in Logistic Regression')
plt.xticks(rotation = 45)
plt.show()



X_train_sm = sm.add_constant(X_train)

logit_model = sm.Logit(Y_train, X_train_sm)
results = logit_model.fit()
print(results.summary())