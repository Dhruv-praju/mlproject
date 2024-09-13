# code to transform the data
import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class DataTransformationConfig:
    '''This class is to specify preprocessing path so that after preprocessing, tranformed data can be stored in .pkl file'''
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    '''This function is responsible for data transformation '''
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # get all numerical and categorical variables
            numerical_columns = ['writing score', 'reading score']
            categorical_columns = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'
            ]

            # define transformation steps on numerical variables
            num_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="median")), # handle missing values by replacing it with median
                    ('scaler', StandardScaler())

                ]
            )
            logging.info('numerical columns encoding completed')

            # define tranformations steps on categorical variables
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info('categorical columns encoding completed')

            # define a preprocessor object that does all the column transformations
            preprocessor = ColumnTransformer(
                                [
                                    ('num_pipeline', num_pipeline, numerical_columns),
                                    ('cat_pipeline',cat_pipeline, categorical_columns)
                                ]
                            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # get train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            # get preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Make X & Y of train and test data
            target_column_name='math score'

            input_train_df=train_df.drop(columns=[target_column_name], axis=1)
            target_train_df=train_df[target_column_name]

            input_test_df=test_df.drop(columns=[target_column_name], axis=1)
            target_test_df=test_df[target_column_name]

            logging.info('Applying preprocessing object on training and testing data')
            
            # Apply transformation to X of train and test data
            input_train_transformed = preprocessing_obj.fit_transform(input_train_df)
            input_test_transformed = preprocessing_obj.transform(input_test_df) 

            # combine trasformed X & y and return the data
            train_arr = np.c_[
                input_train_transformed, np.array(target_train_df)
            ]
            test_arr = np.c_[
                input_test_transformed, np.array(target_test_df)
            ]
            
            # save the preprocesser object as .pkl file
            save_object(
                file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr, test_arr, self.data_tranformation_config.preprocessor_obj_file_path
            )

        except Exception as e :
            raise CustomException(e, sys)

