import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = "artifacts\model.pkl"
            prepocessor_path = "artifacts\preprocessor.pkl"

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=prepocessor_path)


            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)
        

class CustomData:
    def __init__(self,
                 location:str,
                 total_sqft:int,
                 bath:int,
                 bhk:int
                 ):
        self.location = location
        self.total_sqft = total_sqft
        self.bath = bath
        self.bhk = bhk

    def get_data_as_data_frame(self):
        try:
            Custom_data_input_dict = {
                "location":[self.location],
                "total_sqft":[self.total_sqft],
                "bath":[self.bath],
                "bhk":[self.bhk]
            }
            return pd.DataFrame(Custom_data_input_dict)
        except Exception as e:
            raise CustomData(e,sys)    
          

