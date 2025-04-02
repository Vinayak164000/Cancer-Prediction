import os
import sys
import numpy as np
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from src.exceptions import CustomExceptions
from xgboost import XGBClassifier
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trainer_model_file = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer = ModelTrainerConfig()

    def train_model(self, train_array, test_array):
        try:
            logging.info("split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, : -1],
                train_array[:, -1],
                test_array[:, : -1],
                test_array[:, -1]
            )

            # print(X_train)

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "LogisticRegression" : LogisticRegression(max_iter= 1000),
                "KNeighbors": KNeighborsClassifier(),
                # "XG Boost" : XGBClassifier(),
                # "Cat Boost": CatBoostClassifier(),
                "AdaBoost" : AdaBoostClassifier()
            }

            model_report:dict = evaluate_model(X_train, X_test, y_train, y_test, models)
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            save_object(
                file_path = self.model_trainer.trainer_model_file,
                obj= best_model
            )

            predicted = best_model.predict(X_test)
            y_test = np.array(y_test, dtype=int)
            y_test_pred = np.array(predicted, dtype=int)
            accuracy = accuracy_score(y_test, y_test_pred)
            return accuracy, best_model
        

        except Exception as e:
            raise CustomExceptions(e,sys)
