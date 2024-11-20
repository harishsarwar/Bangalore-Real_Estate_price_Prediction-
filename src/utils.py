import os
import sys
import dill
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_oject(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)