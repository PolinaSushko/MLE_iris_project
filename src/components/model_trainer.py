import os
import sys
from dataclasses import dataclass
import tensorflow as tf
from tensorflow import keras
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trainde_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split train and input data")

            X_train, y_train, X_test, y_test = (train_arr[:, :-1], train_arr[:, -1], test_arr[:, :-1], test_arr[:, -1])

            model = keras.Sequential([
                    keras.layers.Dense(10, input_dim = 4, activation = 'relu', kernel_initializer = 'he_normal', 
                                    kernel_regularizer = keras.regularizers.l2(0.01)),
                    keras.layers.BatchNormalization(),
                    keras.layers.Dropout(0.3),
                    keras.layers.Dense(7, activation = 'relu', kernel_initializer = 'he_normal', 
                                    kernel_regularizer = keras.regularizers.l1_l2(l1 = 0.001, l2 = 0.001)),
                    keras.layers.BatchNormalization(),
                    keras.layers.Dropout(0.3),
                    keras.layers.Dense(5, activation = 'relu', kernel_initializer = 'he_normal', 
                                    kernel_regularizer = keras.regularizers.l1_l2(l1 = 0.001, l2 = 0.001)),
                    keras.layers.Dense(3, activation = 'softmax')
                ])
            
            model.compile(optimizer = tf.optimizers.Adam(),
                          loss = 'sparse_categorical_crossentropy',
                          metrics = ['accuracy'])

            model_accuracy = evaluate_model(X_train, y_train, X_test, y_test, model)

            logging.info("Model has been trained")

            save_object(
                file_path = self.model_trainer_config.trainde_model_file_path, 
                obj = model
            )

            return model_accuracy
        
        except Exception as e:
            raise CustomException(e, sys) 