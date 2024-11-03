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
    """
    Configuration class for setting paths and parameters used during the model training process.

    Attributes:
        trainde_model_file_path (str): The file path where the trained model will be saved.
    """
    trainde_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    """
    Prepares the model architecture, compiles it, trains it on the provided training data, evaluates its performance, and saves the trained model.

    Attributes:
        model_trainer_config (ModelTrainerConfig): Configuration instance that holds file paths and settings 
                                                   for model saving.
    """
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        """
        Initiates model training by constructing, compiling, training, and evaluating a neural network model.
        The model is saved upon successful training and evaluation.

        The architecture includes:
            - Dense layers with ReLU activation for learning feature representations.
            - Batch normalization and dropout layers to reduce overfitting.
            - Regularization to prevent model overfitting.
            - A softmax output layer for multi-class classification.

        Args:
            train_arr (np.ndarray): Numpy array containing training data and labels.
            test_arr (np.ndarray): Numpy array containing testing data and labels.

        Returns:
            model_accuracy (float): The accuracy score of the model on the test dataset.
        """
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