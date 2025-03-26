import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import os
from src.exceptions import CustomExceptions
from src.logger import logging
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", 'preprocessor.pkl')


class DataTransformation:
    def __init__(self,):
        self.data_transformation_config = DataTransformationConfig()
        
    def transform(self):

        columns_to_transform = ['Num of pregnancies', 'Smokes', 'Smokes (years)', 'Smokes (packs/year)',
                                'Hormonal Contraceptives','First sexual intercourse','Number of sexual partners']
        
        impute_pipeline = Pipeline(
            steps=[
                ("imputer",SimpleImputer(strategy="most_frequent"))
                ]
        )

        logging.info(f"Columns to impute: {columns_to_transform}")

        preprocessor = ColumnTransformer(
            [
                ('impute_pipeline', impute_pipeline, columns_to_transform)
            ]
        )

        return preprocessor


    def data_transform(self, train_path, test_path):
        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Training and testing data completed")

            preprocessing_obj = self.transform()

            cols = ['IUD','IUD (years)','STDs','STDs (number)','STDs:condylomatosis','STDs:cervical condylomatosis',
                    'STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis','STDs:syphilis',
                    'STDs:pelvic inflammatory disease','STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS',
                    'STDs:HIV','STDs:Hepatitis B','STDs:HPV','STDs: Time since first diagnosis','STDs: Time since last diagnosis',
                    'Hormonal Contraceptives (years)']

            for i in cols:
                train_df.drop(i, axis = 1, inplace = True)
                test_df.drop(i, axis = 1, inplace = True)


            columns_to_transform = ['Num of pregnancies', 'Smokes', 'Smokes (years)', 'Smokes (packs/year)',
                                'Hormonal Contraceptives','First sexual intercourse','Number of sexual partners']
            
            for i in columns_to_transform:
                train_df.replace({'?': train_df[i].mode()[0]}, inplace= True)
                test_df.replace({'?': test_df[i].mode()[0]}, inplace= True)
            target_column = 'Biopsy'
            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]

            logging.info("Applying preprcoessing on trainig and testing dataset")

            train_input_feature = preprocessing_obj.fit_transform(input_feature_train_df)
            test_input_feature = preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[train_input_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[test_input_feature, np.array(target_feature_test_df)]

            logging.info("Saving preprocessed object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomExceptions(e, sys)