import logging
import os
from src.data_loader import DataLoader
from src.preprocessing import SentimentPreprocessor
from src.sentiment_model import SentimentModel
from src.evaluation import ModelEvaluator

logger = logging.getLogger(__name__)

class SentimentPipeline:
    """
    Orchestrates the data loading, preprocessing, training, and evaluation for RoBERTa.
    """
    def __init__(self, config_path: str = "config/config_processed.yaml"):
        self.loader = DataLoader(config_path)
        self.preprocessor = SentimentPreprocessor()
        
        # Load model params from config
        model_params = self.loader.config.get('model_params', {})
        model_name = model_params.get('model_name', "cardiffnlp/twitter-roberta-base-sentiment-latest")
        
        self.model = SentimentModel(model_name=model_name)
        self.config_path = config_path

    def run_training_pipeline(self):
        """
        Full training logic for transformer model.
        """
        logger.info("Starting Pipeline Execution (RoBERTa)...")
        
        # 1. Load Data
        train_df, val_df = self.loader.get_train_val_split()
        
        # 2. Preprocess Data (Minimal cleaning for Transformers)
        logger.info("Cleaning training data text...")
        train_texts = self.preprocessor.preprocess_series(train_df[self.loader.text_col].tolist())
        
        logger.info("Cleaning validation data text...")
        val_texts = self.preprocessor.preprocess_series(val_df[self.loader.text_col].tolist())
        
        # 3. Train Model (Fine-tuning)
        X_train = train_texts
        y_train = train_df[self.loader.target_col].tolist()
        
        X_val = val_texts
        y_val = val_df[self.loader.target_col].tolist()
        
        # Extract training arguments from config
        model_params = self.loader.config.get('model_params', {})
        training_args = {
            "output_dir": "./results",
            "per_device_train_batch_size": model_params.get("batch_size", 16),
            "per_device_eval_batch_size": model_params.get("batch_size", 16),
            "learning_rate": float(model_params.get("learning_rate", 2e-5)),
            "num_train_epochs": model_params.get("epochs", 3),
            "weight_decay": model_params.get("weight_decay", 0.01),
            "logging_steps": model_params.get("logging_steps", 50),
            "eval_steps": model_params.get("eval_steps", 100),
            "eval_strategy": "steps",
            "load_best_model_at_end": True,
            "save_total_limit": 2,
            "report_to": "none"  # Disable wandb/comet etc. by default
        }
        
        self.model.train(X_train, y_train, X_val=X_val, y_val=y_val, training_args_dict=training_args)
        
        # 4. Evaluate Model
        logger.info("Running evaluation on validation set...")
        y_pred = self.model.predict(X_val)
        
        # Ensure labels are in the same format
        labels = sorted(list(set(y_train)))
        eval_results = ModelEvaluator.evaluate(y_val, y_pred, labels=labels)
        
        # 5. Save Model
        model_dir = self.loader.config['paths'].get('model_save_dir', 'models/roberta_sentiment')
        self.model.save(model_dir)
        
        logger.info(f"Pipeline Execution Finished. Model saved to {model_dir}")
        return eval_results

    def predict(self, texts: list):
        """
        Inference logic.
        """
        cleaned_texts = self.preprocessor.preprocess_series(texts)
        return self.model.predict(cleaned_texts)
