import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    """
    Handles the loading of the model and preprocessor, applying transformations to the input features, and predicting the output.
    """
    def __init__(self):
        pass

    def predict(self, features):
        """
        Loads the pre-trained model and preprocessor, scales the input features, and predicts the class labels.

        Args:
            features (pd.DataFrame): A DataFrame with input features for prediction.

        Returns:
            np.ndarray: An array with the predicted class labels.
        """
        try:
            model_path        = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'

            model        = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)

            data_scaled = preprocessor.transform(features)

            preds      = model.predict(data_scaled)
            y_pred_max = np.argmax(preds, axis = 1)

            return y_pred_max
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    """
    Allows the user to specify individual feature values for the iris dataset and provides a method to format these values into a DataFrame compatible with the prediction pipeline.

    Attributes:
        sepal_length (float): The length of the sepal in cm.
        sepal_width (float): The width of the sepal in cm.
        petal_length (float): The length of the petal in cm.
        petal_width (float): The width of the petal in cm.
    """
    def __init__(self,
                 sepal_length : float,
                 sepal_width : float,
                 petal_length : float,
                 petal_width : float
                 ):
        self.sepal_length = sepal_length    
        self.sepal_width  = sepal_width  
        self.petal_length = petal_length
        self.petal_width  = petal_width

    def get_data_as_dataframe(self):
        """
        Creates a DataFrame with the input feature values for prediction. 

        Returns:
            pd.DataFrame: A DataFrame containing feature columns for prediction.
        """
        try:
            custom_data_input_duct = {
                "sepal length (cm)" : [self.sepal_length],
                "sepal width (cm)"  : [self.sepal_width],
                "petal length (cm)" : [self.petal_length],
                "petal width (cm)"  : [self.petal_width]
            }

            return pd.DataFrame(custom_data_input_duct)

        except Exception as e:
            raise CustomException(e, sys)

