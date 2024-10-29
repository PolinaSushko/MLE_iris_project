import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obg(self):
        try:
            num_cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

            preprocessor = ColumnTransformer([
                ('scaler', StandardScaler(), num_cols)  
            ])

            logging.info("Numerical columns standard scaling completed ")

            return preprocessor
        
        except Exception as e:
            logging.error("Error in data transformation", exc_info = True)
            raise e
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)

            logging.info("Reading train and test data is completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_obg()

            target_col = 'target'

            X_train = train_df.drop(columns = [target_col], axis = 1)
            y_train = train_df[target_col]
            X_test  = test_df.drop(columns = [target_col], axis = 1)
            y_test  = test_df[target_col]

            logging.info("Applying preprocessing object on train and test datasets")

            X_train_scaled = preprocessing_obj.fit_transform(X_train)
            X_test_scaled  = preprocessing_obj.transform(X_test)

            train_arr_scaled = np.c_[X_train_scaled, np.array(y_train)]
            test_arr_scaled  = np.c_[X_test_scaled, np.array(y_test)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr_scaled, test_arr_scaled,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
