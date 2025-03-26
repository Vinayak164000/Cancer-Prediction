import os
import sys

import numpy as np
import pandas as pd
import pickle
import dill
from src.exceptions import CustomExceptions
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomExceptions(e, sys)
    
def evaluate_model(x_train, x_test, y_train, y_test, models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]

            y_train = y_train.astype(int) 
            model.fit(x_train, y_train)

            y_test_pred = model.predict(x_test)
            y_test = y_test.astype(int)
            y_test = np.array(y_test, dtype=int)
            y_test_pred = np.array(y_test_pred, dtype=int)

            # test_f1_score = f1_score(y_test, y_test_pred, average="binary")
            # test_recall_score = recall_score(y_test, y_test_pred, average="binary")
            # test_precision_score = precision_score(y_test, y_test_pred, average="binary")
            test_accuracy_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_accuracy_score
            
        return report

    except Exception as e:
        raise CustomExceptions(e, sys)