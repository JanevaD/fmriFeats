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
import seaborn as sns
#%%
data = functional_datasets[0]
data = data['data']
nident = []
features = []
for i, key in enumerate (data):
    nident.append(key)
    feats_name=[]
    features_data = []
    for j, key2 in enumerate(data[key]):
        feats_name.append(key2)
        features_data.append(data[key][key2])
    features.append(features_data)
    


filtered_features = []

for sublist in features:
    filtered_sublist = []
    # Skip the first element of each sublist
    for element in sublist[1:]:
        # Check if the element is a Series
        if isinstance(element, pd.Series):
            # Extract numerical values from the Series
            filtered_sublist.extend([x for x in element.values if isinstance(x, (int, float))])
        # Check if the element is a numeric type directly (int or float)
        elif isinstance(element, (int, float)):
            filtered_sublist.append(element)
    
    filtered_features.append(filtered_sublist)
dataset =pd.DataFrame(np.array(filtered_features), index = nident )

df = pd.merge(dataset, clusters, left_index=True, right_index=True, how='inner')
#%%
X = df.iloc[:,:-1]
Y = df['Clusters']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

model = LogisticRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print (classification_report(Y_test, Y_pred))

confusion_mat = confusion_matrix(Y_test, Y_pred)
sns.heatmap(confusion_mat)
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


