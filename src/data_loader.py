import pandas as pd
import yaml
import os
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles loading and basic cleaning of the Twitter sentiment dataset.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.columns = self.config['data_loader']['columns']
        self.target_col = self.config['data_loader']['target_column']
        self.text_col = self.config['data_loader']['text_column']
        
        logger.info("DataLoader initialized with config.")

    def load_data(self, data_type: str = "train") -> pd.DataFrame:
        """
        Loads CSV data based on type ('train' or 'val').
        """
        path_key = 'train_data' if data_type == "train" else 'val_data'
        file_path = self.config['paths'][path_key]
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Could not find dataset at {file_path}")

        logger.info(f"Loading {data_type} data from {file_path}")
        
        # Determine header and name settings
        has_header = self.config['data_loader'].get('has_header', False)
        header_val = 0 if has_header else None
        names_val = None if has_header else self.columns

        try:
            df = pd.read_csv(file_path, names=names_val, header=header_val, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, names=names_val, header=header_val, encoding='ISO-8859-1')

        # Basic cleaning: drop rows with missing text or target
        initial_len = len(df)
        df = df.dropna(subset=[self.text_col, self.target_col])
        if len(df) < initial_len:
            logger.warning(f"Dropped {initial_len - len(df)} rows with missing values.")

        return df

    def get_train_val_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads both train and validation datasets.
        """
        train_df = self.load_data("train")
        val_df = self.load_data("val")
        return train_df, val_df
