import sys, os
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging


from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function will provide preprocessor object to transform columns
        
        '''
        try:

            target_variable= 'Revenue'
            numerical_columns=['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated', 'ExitRates in %', 'PageValues', 'OperatingSystems', 'Browser', 'Region', 'TrafficType' ]
            categorical_columns= ['Month', 'VisitorType', 'Weekend']

            numerical_pipeline= Pipeline(
                steps=[
                ("scaler",StandardScaler())

                ]
            )

            categorical_pipeline=Pipeline(

                steps=[
                ("ordinal encoder",OrdinalEncoder()),
                ("scaler",StandardScaler())
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",numerical_pipeline,numerical_columns),
                ("cat_pipeline",categorical_pipeline, categorical_columns)
 

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_variable= 'Revenue'
            
            train_feature_df=train_df.drop(columns=[target_variable],axis=1)
            train_target_df=train_df[target_variable]

            test_feature_df=test_df.drop(columns=[target_variable],axis=1)
            test_target_df=test_df[target_variable]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            train_feature_arr=preprocessing_obj.fit_transform(train_feature_df)
            test_feature_arr=preprocessing_obj.transform(test_feature_df)

            train_target_arr = train_target_df.map({True: 1, False: 0})
            test_target_arr = test_target_df.map({True: 1, False: 0})

            train_arr = np.c_[
                train_feature_arr, np.array(train_target_arr)
            ]
            test_arr = np.c_[test_feature_arr, np.array(test_target_arr)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)