# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 12:10:08 2020

@author: fedir
"""




from Model_Gen import temp_name
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


df1 = pd.read_csv('C:/Users/fedir/Documents/Stage/Competition3/X_train_62_14.csv',sep=';')
df1=df1.dropna()
df2 = pd.read_csv('C:/Users/fedir/Documents/Stage/Competition3/y_train_62_14.csv',sep=';')
df2=df2.dropna()
df3 = pd.read_csv('C:/Users/fedir/Documents/Stage/Competition3/X_test_62_14.csv',sep=';')
df3=df3.dropna()


X = df1.values
y = df2.values.ravel()
X_test= df3.values


'''
scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)
X_test = scaler.transform(X_test)

from sklearn.decomposition import PCA
pca = PCA(.95)
pca.fit(X)
#The only transformation done is on the X the y is on scale
X = pca.transform(X)
X_test = pca.transform(X_test)
'''





combination = ['Ridge', 'RandomForestRegressor', 'Lasso', 'MLPRegressor']
test_model = temp_name(base_models = combination)
test_model.fit(X,y) 


#SAVING

y_test_pred = test_model.predict(X_test)

from numpy import savetxt
savetxt('y_pred_fedi_riahi_4.csv',y_test_pred ,header = 'y')


