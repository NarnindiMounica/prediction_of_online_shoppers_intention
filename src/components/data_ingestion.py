import os, sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException

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
    data_ingestion_obj.initiate_data_ingestion()
    
    