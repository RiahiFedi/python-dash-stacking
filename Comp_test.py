# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 12:56:10 2020

@author: fedir
"""


from Model_Gen import temp_name
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import itertools
from sklearn.decomposition import PCA
from datetime import timedelta
from datetime import datetime


def trials(X,y,pca_ = False,t_delta_minutes = 30):
    
    wait_until = datetime.now() + timedelta(minutes=t_delta_minutes)
    if pca_ :
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        
        pca = PCA(.95)
        pca.fit(X)
        X = pca.transform(X)


    # Data split
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.25,shuffle = False)


    model_names = ['Ridge','Lasso','LGBMRegressor',
     'GradientBoostingRegressor','SVR','LinearRegression','MLPRegressor'
     ,'RandomForestRegressor','DecisionTreeRegressor']

    combinations = list()
    for i in range(5,6):
        comb = itertools.combinations(model_names, i)
        combinations += comb
    print(len(combinations))

    df_score = pd.DataFrame(columns = ['r2','rmse','mae','smape','build'])
    for i,c in enumerate(combinations) :   
        test_model = temp_name(base_models = c)
        test_model.fit(X,y) 
        df_score = df_score.append(test_model.eval_model(X_val,y_val))
        df_score= df_score.sort_values(by= ['smape'])
        print(df_score.head(5))
        print(i)
        test_model = None
        if wait_until < datetime.now():
            break
    print(df_score)
    return df_score
