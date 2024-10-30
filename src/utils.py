import os
import sys
import numpy as np
import pandas as pd
import pickle 
from sklearn.metrics import accuracy_score
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, model):
    history = model.fit(X_train, y_train, batch_size = 16, epochs = 500)

    y_pred     = model.predict(X_test)
    y_pred_max = np.argmax(y_pred, axis = 1)

    accuracy = accuracy_score(y_test, y_pred_max)

    return accuracy

def load_object(file_path):
    try:
        with open(file_path, "rb") as f:
             return pickle.load(f)

    except Exception as e:
            raise CustomException(e, sys)