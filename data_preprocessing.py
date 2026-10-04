import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Set up logging to track our pipeline's progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_clean_data(file_path):
    """
    Ingests the CIC-DDoS2019 dataset and performs initial sanitization.
    """
    logging.info(f"Loading dataset from {file_path}...")
    
    # Load the CSV. Low_memory=False prevents mixed data type warnings in large files.
    df = pd.read_csv(file_path, low_memory=False)
    
    logging.info(f"Initial shape: {df.shape[0]} rows, {df.shape[1]} columns.")

    # 1. Strip leading/trailing spaces from column names (common issue in CIC datasets)
    df.columns = df.columns.str.strip()

    # 2. Handle missing and infinite values
    logging.info("Cleaning infinite values and NaNs...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # 3. Standardize the target label (Assuming the label column is named 'Label')
    # We will map 'BENIGN' to 0 and all DDoS variations to 1 for binary classification.
    if 'Label' in df.columns:
        df['Is_Attack'] = df['Label'].apply(lambda x: 0 if x.upper() == 'BENIGN' else 1)
        # Drop the original categorical label
        df = df.drop(columns=['Label'])
    
    logging.info(f"Cleaned shape: {df.shape[0]} rows. Data is ready for splitting.")
    return df

def split_dataset(df, target_col='Is_Attack'):
    """
    Splits the dataset into 70% Train, 15% Validation, and 15% Test sets.
    """
    logging.info("Initiating 70/15/15 data split...")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First, split off 70% for training, leaving 30% for validation/testing
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    
    # Next, split the remaining 30% equally into 15% validation and 15% testing
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    
    logging.info(f"Split Complete.")
    logging.info(f"Training set: {X_train.shape[0]} samples (70%)")
    logging.info(f"Validation set: {X_val.shape[0]} samples (15%)")
    logging.info(f"Testing set: {X_test.shape[0]} samples (15%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# --- Execution Block ---
if __name__ == "__main__":
    # Replace with the actual path to the downloaded CIC-DDoS2019 CSV
    DATA_PATH = "path_to_your_cic_ddos_file.csv" 
    
    try:
        clean_df = load_and_clean_data(DATA_PATH)
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(clean_df)
    except FileNotFoundError:
        logging.error("Dataset not found. Please ensure the path is correct.")