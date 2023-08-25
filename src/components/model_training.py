import os, sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'trained_model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_initiator = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting train and test data')
            x_train, y_train, x_test, y_test = (train_array[:,:-1], train_array[:,-1], test_array[:, :-1], test_array[:, -1])

            models = {'logistic_regression':LogisticRegression(),
            'decision_tree_classifier': DecisionTreeClassifier(),
            'random_forest_classifier': RandomForestClassifier(),
            'k_neighbors_classifier': KNeighborsClassifier(),
            'adaboost_classifier': AdaBoostClassifier(),
            'gradient_boosting_classifier': GradientBoostingClassifier(),
            'xgb_classifier': XGBClassifier()}

            model_report = evaluate_models(x_train,y_train, x_test, y_test, models)

            # To get best model score and model name from report dictionary

            best_model_score = max(model_report.values())
            best_model_name = max(zip(model_report.values(), model_report.keys()))[1]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found')

            logging.info('Best model is found on training and testing data')   

            save_object(file_path = self.model_trainer_initiator.trained_model_file_path,
            obj= best_model) 

            predicted = best_model.predict(x_test)
            best_roc_auc_score = roc_auc_score(y_test, predicted)

            logging.info('Best model roc_auc_score is calculated') 

            return best_roc_auc_score, best_model_name
            
        except Exception as e:
            raise CustomException(e,sys)




