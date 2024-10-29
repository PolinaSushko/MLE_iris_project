import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    """
    Defines the paths used to save the raw, training, and testing datasets 
    during the data ingestion process. The files are stored in the 'artifacts' directory 
    to keep all generated data files organized.

    Attributes:
        train_data_path (str): File path for the training data CSV.
        test_data_path (str): File path for the test data CSV.
        raw_data_path (str): File path for the raw data CSV.
    """
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    """
    This class is responsible for loading the dataset, splitting it into training and 
    test sets, and saving them to designated file paths specified in `DataIngestionConfig`.

    Methods:
        initiate_data_ingestion: Loads a dataset (Iris dataset in this case), splits it 
                                 into train and test sets, and saves them to CSV files.
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process.
        Loads the Iris dataset, converts it into a DataFrame, splits the data into training 
        and test sets, and saves these sets to specified file paths.

        Returns:
            tuple: Paths to the saved training and test datasets.
        """
        logging.info("Entered the data ingestion method or component")

        try:
            iris_sklearn = load_iris()
            df = pd.concat([
                pd.DataFrame(iris_sklearn['data'], columns = [iris_sklearn.feature_names[i] for i in range(iris_sklearn.data.shape[1])]),
                pd.DataFrame(iris_sklearn['target'], columns = ['target'])
                ], axis = 1)
            
            logging.info("Read dataset as a dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)

            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

            logging.info("Train test split initiated")

            train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42, shuffle = True)

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Ingestion of data is completed")

            return (
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
            #logging.error("Error in data ingestion", exc_info = True)
            #raise e
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)