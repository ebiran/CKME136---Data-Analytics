# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 02:05:30 2018

@author: ERANA
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score


df = pd.read_csv('C:/Users/ERANA/Desktop/clean_data.csv')

dff = df[['Elevation','MeanTemp','HeatDegDays','Total_Precip_mm']]
dff.describe()
#Box plot
dff.boxplot()
#Hist plot
dff.hist()
corr_matrix = df.corr()
corr_matrix['Total_Precip_mm'].sort_values(ascending=False)

#Scatter by Total Precipitation
from pandas.tools.plotting import scatter_matrix
df.plot(kind="scatter", x='Longitude', y= 'Latitude', alpha=0.4, figsize=(10,7),
    c= df.Total_Precip_mm, cmap=plt.get_cmap("jet"), colorbar=True,sharex=False)


#Linear Regression Plot
dfl = pd.read_csv('C:/Users/ERANA/Desktop/clean_data.csv')

Mean = dfl.groupby(['Month','Day'])['MeanTemp'].mean()
Rain = dfl.groupby(['Month','Day'])['Total_Precip_mm'].mean()

Mean = Mean.values[:,np.newaxis]
Rain = Rain.values

modell = LinearRegression()
modell.fit(Mean, Rain)
plt.scatter(Mean, Rain,color='r')
plt.plot(Mean, modell.predict(Mean),color='k')
plt.show()

#number of stations
len(df['Climate_ID'].value_counts())

df = df[['Elevation','Year','Month','Day','MeanTemp','HeatDegDays','Total_Precip_mm']]
df.head()

X = df[['Elevation','Year','Month','Day','Total_Precip_mm','HeatDegDays']]
y = df['MeanTemp']


#Train a Linear Regression
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print('Linear Regression R squared": %.3f' % regressor.score(X_test, y_test))

# Use cross validation
scores = cross_val_score(regressor, X, y, cv=5)
print(scores)
print('average score: {}'.format(scores.mean()))

#root-mean-square error (RMSE)
from sklearn.metrics import mean_squared_error
lin_mse = mean_squared_error(y_pred, y_test)
lin_rmse = np.sqrt(lin_mse)
print('Linear Regression RMSE: %.3f' % lin_rmse)


#Mean absolute error (MAE):

from sklearn.metrics import mean_absolute_error
lin_mae = mean_absolute_error(y_pred, y_test)
print('Linear Regression MAE: %.3f' % lin_mae)

#Decision Tree Model
from sklearn import tree
from sklearn.metrics import accuracy_score
df = pd.read_csv('C:/Users/ERANA/Desktop/clean_data.csv')
df1 = df[['Elevation','Year','Month','Day','MeanTemp','HeatDegDays','Total_Precip_mm']]
X = df1.drop(['MeanTemp'], axis=1)
y = np.array(df1['MeanTemp'],dtype='long')
#Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#fit the model
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
#Accuracy Score:
print('Accuracy: ', (accuracy_score(y_test, y_predict) * 100) ,'%')


#Gradient boosting (Ensemble Method)
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
model = ensemble.GradientBoostingRegressor()
model.fit(X_train, y_train)
print('Gradient Boosting R squared": %.3f' % model.score(X_test, y_test))

y_pred = model.predict(X_test)
model_mse = mean_squared_error(y_pred, y_test)
model_rmse = np.sqrt(model_mse)
print('Gradient Boosting RMSE: %.3f' % model_rmse)

#Feature Importance
feature_labels = np.array(['Total_Precip_mm','Elevation','MeanTemp','Latitude','Longitude','HeatDegDays'])
importance = model.feature_importances_
feature_indexes_by_importance = importance.argsort()
for index in feature_indexes_by_importance:
    print('{}-{:.2f}%'.format(feature_labels[index], (importance[index] *100.0)))