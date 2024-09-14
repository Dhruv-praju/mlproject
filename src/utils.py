# common code to entire application goes here

import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException

from sklearn.metrics import r2_score

def save_object(file_path, obj):
    '''This function saves the given obj as a file'''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models:dict):
    '''This fuction evaluates the given models and returns the report'''
    try:
        report = dict()

        for algo in models.keys():
            model = models[algo]

            # train the model
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate Train and Test
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # add r2 to the report
            report[algo] = test_model_score

        return report
            
    except Exception as e:
        raise CustomException(e, sys)