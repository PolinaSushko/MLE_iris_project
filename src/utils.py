import os
import sys
import numpy as np
import pandas as pd
import pickle 
from sklearn.metrics import accuracy_score
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Saves an object to a specified file path using pickle. 

    Args:
        file_path (str): The file path where the object should be saved.
        obj (object): The object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, model):
    """
    Trains the model using the training data, predicts the test data labels, and calculates the accuracy score.

    Args:
        X_train (np.ndarray): The training data features.
        y_train (np.ndarray): The training data labels.
        X_test (np.ndarray): The test data features.
        y_test (np.ndarray): The test data labels.
        model (keras.Model): The neural network model to train and evaluate.

    Returns:
        float: The accuracy score of the model on the test set.
    """
    history = model.fit(X_train, y_train, batch_size = 16, epochs = 500)

    y_pred     = model.predict(X_test)
    y_pred_max = np.argmax(y_pred, axis = 1)

    accuracy = accuracy_score(y_test, y_pred_max)

    return accuracy

def load_object(file_path):
    """
    Retrieves a previously saved object for reuse, such as a trained model or preprocessor.

    Args:
        file_path (str): The file path where the object is saved.

    Returns:
        object: The loaded object.
    """
    try:
        with open(file_path, "rb") as f:
             return pickle.load(f)

    except Exception as e:
            raise CustomException(e, sys)