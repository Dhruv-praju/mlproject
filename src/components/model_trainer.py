# code for training

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from dataclasses import dataclass
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models


@dataclass
class ModelTrainerConfig:
    '''class to specify trained model path'''
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        '''This function trains all the models, finds the best one and saves it'''
        try:
            # split train and test data to X_train, y_train .....
            logging.info('Split training and test input data')
            X_train, X_test, y_train, y_test = ( train_array[:, :-1], test_array[:, :-1], train_array[:,-1], test_array[:,-1] )

            models = {
                "Linear Regression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "Support Vector":SVR(),
                "K-NeighborsRegressor":KNeighborsRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Random Forest":RandomForestRegressor(),
                "AdaBoost":AdaBoostRegressor(),
                "CatBoost":CatBoostRegressor(verbose=False),
                "Gradient Boost": GradientBoostingRegressor(),
                "XG Boost":XGBRegressor()
            }
            params = {
                "Linear Regression":{},
                "Ridge":{},
                "Lasso":{},
                "Support Vector":{},
                "K-NeighborsRegressor":{
                    "n_neighbors": [2,5, 10, 20, 50]
                },
                "Decision Tree":{
                        'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                        # 'splitter':['best','random'],
                        # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                        "max_depth":[5,8,15, None, 10], # depth of the tree
                        "max_features": [5,7, "auto", 8], # Maximum features to look for best split node
                        "min_samples_split": [2, 8, 15, 20], # Minimum samples need to split node
                        "n_estimators": [100, 200, 500, 1000]  # No. of Decision trees
                },
                "AdaBoost":{
                        "n_estimators":[50, 80, 100, 150]                       
                },
                "CatBoost":{
                        'depth': [6,8,10],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'iterations': [30, 50, 100]
                            },
                "Gradient Boost":{
                        "n_estimators":[50, 100, 150, 200, 300],
                        "learning_rate": [0.01, 0.1, 1],
                        "max_depth": [3, 5, 8, 10],
                        "loss": ['squared_error', 'absolute_error', 'huber', 'quantile'],
                        "criterion": ['friedman_mse', 'squared_error'],
                        "min_samples_split":[2, 8, 10],
                        "min_impurity_decrease":[0, 0.01, 0.05]
                },
                "XG Boost":{
                        'learning_rate':[.1,.01,.05,.001],
                        'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict = evaluate_models(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, models=models, params=params )
            
            # get the best model score and name from the dict
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info('Best found model on both training and testing dataset')
            
            # save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)

            return r2

        except Exception as e:
            raise CustomException(e)