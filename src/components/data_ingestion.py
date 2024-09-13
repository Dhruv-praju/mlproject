## code to read the data from data source like mongoDB or Hadoop or API,etc
import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    '''class to specify training testing paths so that train data and test data can be stored'''
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # instanciate dataIngestionConfig object

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method or component")
        try:
            # read the data from csv file or mongoDB
            df = pd.read_csv('src/notebooks/data/StudentsPerformance.csv')
            logging.info("Read the dataset as dataframe")

            # create directories of train data path
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # save raw data to the directory
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")

            # get train and test data 
            train_set,test_set = train_test_split(df, test_size=0.2, random_state=42)
            # save it to the directory
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data is completed")

            # return train and test data path
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
     obj = DataIngestion()
     train_data, test_data = obj.initiate_data_ingestion()

     data_transformation = DataTransformation()
     data_transformation.initiate_data_transformation(train_data, test_data)