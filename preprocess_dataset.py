import pandas as pd
import os
import logging
from pathlib import Path

# --- CONFIGURATION ---
INPUT_DIR = "dataset"
OUTPUT_DIR = "processed_dataset"
TRAIN_FILE = "twitter_training.csv"
VAL_FILE = "twitter_validation.csv"

# Column indices to keep (0-indexed)
# 0: ID, 1: Entity, 2: Sentiment, 3: Text
# We remove 0 and 1, keeping 2 and 3.
COLUMNS_TO_KEEP = [2, 3]
COLUMN_NAMES = ["sentiment", "text"]

# --- SETUP LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def preprocess_csv(file_path, output_path):
    """
    Cleans a Twitter Sentiment CSV by removing the first two columns
    and saving only the Sentiment and Text columns.
    """
    try:
        logger.info(f"Processing: {file_path}")
        
        # Load CSV without headers (original files don't have them)
        df = pd.read_csv(file_path, header=None)
        
        # Select specified columns and assign names
        # Some rows might have missing text, so we handle that if needed
        df_cleaned = df.iloc[:, COLUMNS_TO_KEEP]
        df_cleaned.columns = COLUMN_NAMES
        
        # Basic cleaning: Drop rows with completely empty text
        initial_count = len(df_cleaned)
        df_cleaned = df_cleaned.dropna(subset=["text"])
        final_count = len(df_cleaned)
        
        if initial_count != final_count:
            logger.warning(f"Dropped {initial_count - final_count} rows with missing text in {file_path}")

        # Save to the new directory
        df_cleaned.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Successfully saved {final_count} rows to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")

def main():
    # 1. Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logger.info(f"Created directory: {OUTPUT_DIR}")

    # 2. Define files to process
    files_to_process = [
        (Path(INPUT_DIR) / TRAIN_FILE, Path(OUTPUT_DIR) / f"cleaned_{TRAIN_FILE}", "Training Data"),
        (Path(INPUT_DIR) / VAL_FILE, Path(OUTPUT_DIR) / f"cleaned_{VAL_FILE}", "Validation Data")
    ]
    
    from tqdm import tqdm
    # 3. Process each file with progress bar
    for input_path, output_path, desc in tqdm(files_to_process, desc="Processing Datasets"):
        if input_path.exists():
            logger.info(f"Starting {desc}...")
            preprocess_csv(input_path, output_path)
        else:
            logger.error(f"{desc} file NOT found at {input_path}")

    logger.info("Dataset preprocessing complete.")

if __name__ == "__main__":
    main()
