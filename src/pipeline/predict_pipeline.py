import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        '''It predicts the output of given new datapoint'''
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/preprocessor.pkl'

            # load model and preprocessor file
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # predict the feature
            data_preprocessed = preprocessor.transform(features)
            preds = model.predict(data_preprocessed)

            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    '''This class is responsible for mapping all input given by the user to backend'''
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_input_as_dataframe(self):
        '''Creates Dataframe of input data and returns it'''
        try:
            input_data_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score],
            }

            return pd.DataFrame(input_data_dict)
        except Exception as e:
            raise CustomException(e, sys)
            
