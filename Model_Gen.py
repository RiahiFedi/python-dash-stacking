# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 08:40:39 2020

@author: Fedi
"""
import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.base import clone
from lightgbm import LGBMRegressor
import pickle


import random

['KNeighborsRegressor','Ridge','LGBMRegressor','DecisionTreeRegressor','Lasso',
 'GradientBoostingRegressor','SVR','LinearRegression','MLPRegressor','RandomForestRegressor']




class temp_name:
    

    def select_models(self,n):
        if n == None : 
            n = 6
        model_names = ['knn','cart','SVR','lr','NN','RF','GBR','Lasso']
        model_names = random.sample(model_names, n)
        
        return model_names

    
    def get_models(self,model_names,hyperparameters = None):  

        model_list = dict()
        i = 0
        for i in model_names:
            if (i == 'KNeighborsRegressor'):
                model_list['KNeighborsRegressor'] = KNeighborsRegressor()
                if hyperparameters!=None:
                    model_list['KNeighborsRegressor'].set_params(**hyperparameters['KNeighborsRegressor'])
            
            elif (i == 'Ridge'):
                model_list['Ridge'] = Ridge()
                if hyperparameters!=None:
                    model_list['Ridge'].set_params(**hyperparameters['Ridge'])
                    
            elif (i == 'LGBMRegressor'):
                model_list['LGBMRegressor'] = LGBMRegressor()
                if hyperparameters!=None:
                    model_list['LGBMRegressor'].set_params(**hyperparameters['LGBMRegressor'])
                    
            elif (i == 'DecisionTreeRegressor'):
                model_list['DecisionTreeRegressor'] = DecisionTreeRegressor(random_state=0)
                if hyperparameters!=None:
                    model_list['DecisionTreeRegressor'].set_params(**hyperparameters['DecisionTreeRegressor'])
                    
            elif (i == 'Lasso'):
                model_list['Lasso'] = Lasso()
                if hyperparameters!=None:
                    model_list['Lasso'].set_params(**hyperparameters['Lasso'])
                    
            elif (i == 'GradientBoostingRegressor'):
                model_list['GradientBoostingRegressor'] = GradientBoostingRegressor(random_state=0)
                if hyperparameters!=None:
                    model_list['GradientBoostingRegressor'].set_params(**hyperparameters['GradientBoostingRegressor'])
                    
            elif (i == 'SVR'):
                model_list['SVR'] = SVR(gamma='auto')
                if hyperparameters!=None:
                    model_list['SVR'].set_params(**hyperparameters['SVR'])
                    
                    
            elif (i == 'LinearRegression'):
                model_list['LinearRegression'] = LinearRegression()
                
            elif (i == 'MLPRegressor'):
                model_list['MLPRegressor' ] = MLPRegressor()
                if hyperparameters!=None:
                    model_list['MLPRegressor'].set_params(**hyperparameters['MLPRegressor'])
                    
            elif(i == 'RandomForestRegressor'):
                model_list['RandomForestRegressor'] = RandomForestRegressor()
                if hyperparameters!=None:
                    model_list['RandomForestRegressor'].set_params(**hyperparameters['RandomForestRegressor'])
                    
                    
    
        #model_list['XGB'] = xgb.XGBRegressor(objective="reg:linear", random_state=42)
    
        return model_list    
    
    def __init__(self,hyperparameters = None,n_base=None,
                 n_folds = None,base_models = None,meta_model = None):

        if n_base == None : 
            if base_models == None:
                self.n_base = 4
            else :
                self.n_base = len(base_models) 
        else :
            self.n_base = n_base
            
        if n_folds == None: 
            self.n_folds = 3
        else :
            self.n_folds = n_folds
            
        if base_models == None : 
            self.base_models = self.get_models(self.select_models(n_base))
        elif hyperparameters == None :
            self.base_models = self.get_models(base_models)
        else : 
            self.base_models = self.get_models(base_models,hyperparameters)
            
        if meta_model == None : 
            self.meta_model = LinearRegression()
        else :
             self.meta_model = meta_model
        
        
    

    '''def _Smape(self,A, F):
        return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))'''
    
    
    
    def _Smape(self,y, y_pred):
        denom = np.sum(np.abs(y + y_pred))
        if denom == 0:
            return np.NaN
        num = np.sum(np.abs(y - y_pred))
        res = num/denom
        return res
    
    
    def _create_meta_dataset(self,X, y):

        self.base_models_ = list()
        for x in self.base_models:
            self.base_models_.append(list())
        
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        out_of_fold_predictions = np.zeros((X.shape[0], self.n_base))
        print(out_of_fold_predictions.shape)
        j= 0
        for j, model_name in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                
                instance = clone(self.base_models[model_name])
                self.base_models_[j].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, j] = y_pred
        
               
        #return np.concatenate((X, out_of_fold_predictions), axis=1)
        print(out_of_fold_predictions.shape)
        return out_of_fold_predictions
        
    
    def fit(self,X,y):
        out_of_fold_predictions = self._create_meta_dataset(X,y)
        self.meta_model.fit(out_of_fold_predictions, y)
        return self
    
    
    def predict(self, X):     
        meta_features = np.column_stack([np.column_stack([model.predict(X) for model in base_models]).mean(axis=1) for base_models in self.base_models_ ])
        print(meta_features.shape)
        return self.meta_model.predict(meta_features)

    def eval_model(self,X_val,y_val):
        # evaluate sub models on hold out dataset
        accuracy_df = pd.DataFrame(columns=('r2', 'rmse','smape','build'))
        y_pred_M = self.predict(X_val)

        r2 = round(metrics.r2_score(y_val, y_pred_M),2)
        rmse = round(np.sqrt(metrics.mean_squared_error(y_val, y_pred_M)),2)
        mae = round(metrics.mean_absolute_error(y_val, y_pred_M),2)
        smape = self._Smape(y_val,y_pred_M)*100
        accuracy_df = accuracy_df.append(pd.DataFrame({'r2':[r2],'rmse':[rmse],'mae':[mae],'smape':[smape],'build':[self.base_models.keys()]}))
        print(accuracy_df)
        return pd.DataFrame({'r2':[r2],'rmse':[rmse],'mae':[mae],'smape':[smape],'build':[self.base_models.keys()]})

    
    
    
    def save_models(self, path = ''):

        f= open(path +"structure.txt",'w+')
        for base_models in self.base_models_:
            for i,model in enumerate(base_models):
                filename = path +'Base/'+type(model).__name__ +str(i)+'.sav'
                pickle.dump(model, open(filename, 'wb'))
                print(filename)
                f.write('0'+type(model).__name__+ str(i) +'\n') 
        
        filename = path +'Meta/'+type(self.meta_model).__name__+'.sav'
        print(filename)
        pickle.dump(self.meta_model, open(filename, 'wb'))
        f.write('1'+type(self.meta_model).__name__+'\n') 
        f.close
        
        

    def load_model(self,path = ''):
        #models_n = os.listdir('C:/Users/Said/Project ST/Base')
        #if '.ipynb_checkpoints' in models_n :
        #    models_n.remove('.ipynb_checkpoints')
        #for i in range(len(models_n)):
        #    models_n[i] = models_n[i][:-4]
        #models = base_models(models_n)

        l = list()
        models_n = list()
        f=open(path + "structure.txt", "r")
        f1 = f.readlines()
        for x in f1:
            l.append(x)
        for i in range(len(l)):
            if l[i][0] == '0':
                models_n.append(l[i][1:-1]) 
            elif l[i][0] == '1':
                meta_filename = path+ 'Meta/'+ l[i][1:-1] +'.sav'
                print(meta_filename)
    
        print(models_n)
        
        n_base = 0
        for m in models_n:
            if m[-1]=='0':
                n_base +=1
        n_folds = len(models_n)//n_base
        
        
        #loaded_model = temp_name(n_base = n_base,n_folds = n_folds)        
        self.n_base = n_base
        self.n_folds = n_folds
        
        
        self.base_models_ = list()
        for x in range(n_base):
            self.base_models_.append(list())
            
            
        bm = list()
        for k in range(n_base):
            bm.append(models_n[k*self.n_folds][0:-1])
        self.base_models = self.get_models(bm)

            
        for i in range(n_base):
            for j in range(len(models_n)//n_base):
                filename = path+ 'Base/'+models_n[i*self.n_folds + j] + '.sav'
                self.base_models_[i].append(pickle.load(open(filename, 'rb')))
                print(filename)
                

        self.meta_model = pickle.load(open(meta_filename, 'rb'))
        #return loaded_model