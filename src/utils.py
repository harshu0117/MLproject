import sys
import os

import numpy as np
import pandas as pd
import dill
import pickle

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        
        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")

            # Train the model
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R^2 scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Log the scores
            logging.info(f"{model_name} - Train R2: {train_model_score}, Test R2: {test_model_score}")

            # Store the test score in the report
            report[model_name] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e,sys)