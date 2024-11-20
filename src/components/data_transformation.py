import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils import save_oject
from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ['total_sqft', 'bath', 'bhk']
            categorical_columns = ['location']

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(sparse_output=False)),  # Set sparse_output=False for dense output
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def intiate_data_transformation(self, train_path, test_path):
        try:
            # Load train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Loading train and test data files')

            # Get the preprocessor
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'price'

            # Divide the train dataset into independent and dependent features
            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Divide the test dataset into independent and dependent features
            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('Applying preprocessing on training and test dataframes')

            # Apply transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)

            # Convert the target columns to 2D arrays
            target_feature_train_arr = target_feature_train_df.values.reshape(-1, 1)
            target_feature_test_arr = target_feature_test_df.values.reshape(-1, 1)

            # Concatenate transformed features with target
            train_arr = np.concatenate([input_feature_train_arr, target_feature_train_arr], axis=1)
            test_arr = np.concatenate([input_feature_test_arr, target_feature_test_arr], axis=1)

            logging.info(f"Saved preprocessing object")

            # Save the preprocessor object
            save_oject(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils import save_oject
from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ['total_sqft', 'bath', 'bhk']
            categorical_columns = ['location']

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(sparse_output=False)),  # Set sparse_output=False for dense output
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def intiate_data_transformation(self, train_path, test_path):
        try:
            # Load train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Loading train and test data files')

            # Get the preprocessor
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'price'

            # Divide the train dataset into independent and dependent features
            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Divide the test dataset into independent and dependent features
            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('Applying preprocessing on training and test dataframes')

            # Apply transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)

            # Convert the target columns to 2D arrays
            target_feature_train_arr = target_feature_train_df.values.reshape(-1, 1)
            target_feature_test_arr = target_feature_test_df.values.reshape(-1, 1)

            # Concatenate transformed features with target
            train_arr = np.concatenate([input_feature_train_arr, target_feature_train_arr], axis=1)
            test_arr = np.concatenate([input_feature_test_arr, target_feature_test_arr], axis=1)

            logging.info(f"Saved preprocessing object")

            # Save the preprocessor object
            save_oject(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
