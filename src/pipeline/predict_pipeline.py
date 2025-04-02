import sys
import pandas as pd
from src.exceptions import CustomExceptions
from src.utils import load_object
import numpy as np


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'  # Use forward slashes for cross-platform compatibility
            preprocessor_path = 'artifacts/preprocessor.pkl'
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Convert features to DataFrame if needed
            if isinstance(features, np.ndarray):
                features = pd.DataFrame(features, columns=preprocessor.feature_names_in_)

            # Apply preprocessing
            data_scales = preprocessor.transform(features)

            # Pass transformed features to the model (NOT the original features)
            preds = model.predict(data_scales)
            return preds
        except Exception as e:
            raise CustomExceptions(e, sys)



class CustomData:
    def __init__(self,
                Age : int,
                Number_of_sexual_partners: str,
                First_sexual_intercourse: str,
                Num_of_pregnancies : str,
                Smokes : str,
                Smokes_years:str,
                Smokes_packs_year:str,
                Hormonal_Contraceptives:str,
                STDs_Number_of_diagnosis:int,
                Dx_Cancer:int,
                Dx_CIN:int,
                Dx_HPV:int,
                Dx:int,
                Hinselmann:int,
                Schiller:int,
                Citology:int):
        self.age = Age
        self.sexual_partners = Number_of_sexual_partners
        self.sexual_intercourse = First_sexual_intercourse
        self.num_pregnancies = Num_of_pregnancies
        self.smoke = Smokes
        self.smoke_in_year = Smokes_years
        self.smoke_per_year = Smokes_packs_year
        self.hormonal_contraceptives = Hormonal_Contraceptives
        self.STDs = STDs_Number_of_diagnosis
        self.Cancer = Dx_Cancer
        self.CIN = Dx_CIN
        self.HPV = Dx_HPV
        self.Dx = Dx
        self.Hinselmann = Hinselmann
        self.Schiller = Schiller
        self.Citology = Citology

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age" : [self.age],
                "Number of sexual partners": [self.sexual_partners],
                "First sexual intercourse" : [self.sexual_intercourse],
                "Num of pregnancies": [self.num_pregnancies],
                "Smokes": [self.smoke],
                "Smokes (years)": [self.smoke_in_year],
                "Smokes (packs/year)": [self.smoke_per_year],
                "Hormonal Contraceptives": [self.hormonal_contraceptives],
                "STDs: Number of diagnosis": [self.STDs],
                "Dx:Cancer": [self.Cancer],
                "Dx:CIN": [self.CIN],
                "Dx:HPV": [self.HPV],
                "Dx": [self.Dx],
                "Hinselmann": [self.Hinselmann],
                "Schiller": [self.Schiller],
                "Citology": [self.Citology]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomExceptions(e, sys)