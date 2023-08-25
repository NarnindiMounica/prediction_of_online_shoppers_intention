import os, sys
import dill
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(x_train, y_train, x_test, y_test, models):   
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]

            #training the model
            model.fit(x_train, y_train)

            y_train_prediction = model.predict(x_train)
            y_test_prediction = model.predict(x_test)

            train_score = roc_auc_score(y_train, y_train_prediction)
            test_score = roc_auc_score(y_test, y_test_prediction)

            report[list(models.keys())[i]] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)