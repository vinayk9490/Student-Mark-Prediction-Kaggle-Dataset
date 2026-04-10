import os
import sys
import numpy as np
import pandas as pd
import pickle
import warnings 

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.logger import logging
from src.exception import CustomException


def save_object(file_path: str, obj: object) -> None:
    '''
    The pickel file contains the preprocessor that tells what column is numerical 
    and what column is categorical and how to transform them. 
    This function is responsible for saving the preprocessor object as a pickle file.
    '''
    logging.info("Entered the save_object method of utils")
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict = {}) -> dict:
    '''
    This function is responsible for evaluating the models and returning the report containing the r2 score of each model.
    '''
    logging.info("Entered the evaluate_models method of utils")
    try:
        report: dict = {}

        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            model_params = params.get(model_name, {})

            if model_params:
                warnings.filterwarnings("ignore")
                gs = GridSearchCV(model, model_params, cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str) -> object:
    '''
    This function is responsible for loading the preprocessor object from the pickle file.
    '''
    logging.info("Entered the load_object method of utils")
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)