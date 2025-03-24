from curses import raw
import os
import sys
from turtle import st
from src.exceptions import CustomExceptions
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingeston method or component")
        try:
            df = pd.read_csv('')
        except:
            pass