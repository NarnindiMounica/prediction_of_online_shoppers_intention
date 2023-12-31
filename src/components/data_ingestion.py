import os, sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException

from src.components.data_transformation import DataTransformationConfig, DataTransformation

from src.components.model_training import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts', 'train.csv' )
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered data ingestion stage')  
        try:
            df = pd.read_csv('Notebook\data\online_shoppers_intention.csv')
            logging.info('Read dataset into dataframe')

            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Removing the duplicate rows from data')
            df = df[~df.duplicated()]

            logging.info('Dropping unwanted columns from dataframe')
            df = df.drop(['SpecialDay (probability)', 'ProductRelated_Duration', 'BounceRates in %'], axis =1)

            logging.info('Train and test data split initiated')
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=1)

            train_data.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True )
            logging.info('Data ingestion is completed')

            return (
                self.data_ingestion_config.train_data_path, 
                self.data_ingestion_config.test_data_path
                )

        except Exception as e:
            raise CustomException(e, sys)  


if __name__=='__main__':
    data_ingestion_obj = DataIngestion()
    train_path, test_path = data_ingestion_obj.initiate_data_ingestion()

    data_transformation_obj = DataTransformation()
    train_array, test_array, _=data_transformation_obj.initiate_data_transformation(train_path, test_path)

    model_trainer_obj = ModelTrainer()
    model_name,score=model_trainer_obj.initiate_model_trainer(train_array, test_array)
    print(model_name, score)


    
    