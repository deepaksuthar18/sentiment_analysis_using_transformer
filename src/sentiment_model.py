import os
import logging
import torch
import numpy as np
from typing import Optional, List, Union, Dict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    pipeline,
    DataCollatorWithPadding
)
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

logger = logging.getLogger(__name__)

class SentimentModel:
    """
    Production-grade Sentiment Analysis Model using RoBERTa and Hugging Face Transformers.
    """
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest", 
                 model_dir: Optional[str] = None, 
                 num_labels: int = 3):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.label_encoder = LabelEncoder()
        
        if model_dir and os.path.exists(model_dir):
            logger.info(f"Loading existing model from {model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
            # Try to load label encoder if it exists
            import joblib
            le_path = os.path.join(model_dir, "label_encoder.joblib")
            if os.path.exists(le_path):
                self.label_encoder = joblib.load(le_path)
        else:
            logger.info(f"Initializing new model from {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            ).to(self.device)

    def _tokenize_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True, padding=False, max_length=128)

    def train(self, X_train: List[str], y_train: List[Union[int, str]], 
              X_val: Optional[List[str]] = None, y_val: Optional[List[Union[int, str]]] = None,
              training_args_dict: Optional[Dict] = None):
        """
        Fine-tunes the RoBERTa model using the Hugging Face Trainer API.
        """
        logger.info("Starting model training (fine-tuning)...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_train)
        num_labels = len(self.label_encoder.classes_)
        
        # Update model if num_labels changed
        if num_labels != self.model.config.num_labels:
            logger.info(f"Updating model to handle {num_labels} labels.")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            ).to(self.device)

        # Prepare datasets
        train_dataset = Dataset.from_dict({"text": X_train, "label": y_encoded})
        train_dataset = train_dataset.map(self._tokenize_function, batched=True)
        
        val_dataset = None
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            val_dataset = Dataset.from_dict({"text": X_val, "label": y_val_encoded})
            val_dataset = val_dataset.map(self._tokenize_function, batched=True)

        # Training arguments
        if training_args_dict is None:
            training_args_dict = {
                "output_dir": "./results",
                "eval_strategy": "steps" if val_dataset else "no",
                "learning_rate": 2e-5,
                "per_device_train_batch_size": 16,
                "per_device_eval_batch_size": 16,
                "num_train_epochs": 3,
                "weight_decay": 0.01,
                "save_total_limit": 2,
                "logging_steps": 50,
                "eval_steps": 100,
                "load_best_model_at_end": True if val_dataset else False,
            }
        
        training_args = TrainingArguments(**training_args_dict)
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
        )
        
        trainer.train()
        logger.info("Model training completed.")

    def predict(self, X: List[str], batch_size: int = 32) -> List[str]:
        """
        Predicts labels for a list of texts using batching and tqdm for progress.
        """
        self.model.eval()
        all_predictions = []
        
        from tqdm import tqdm
        for i in tqdm(range(0, len(X), batch_size), desc="Predicting Sentiment"):
            batch_texts = X[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=128
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.extend(predictions)
            
        return self.label_encoder.inverse_transform(all_predictions).tolist()

    def predict_proba(self, X: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Predicts probabilities for class labels using batching and tqdm for progress.
        """
        self.model.eval()
        all_probs = []
        
        from tqdm import tqdm
        for i in tqdm(range(0, len(X), batch_size), desc="Calculating Probabilities"):
            batch_texts = X[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=128
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            all_probs.append(probabilities.cpu().numpy())
            
        return np.vstack(all_probs)

    def save(self, directory: str):
        """
        Saves the model, tokenizer, and label encoder to disk.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        
        # Save LabelEncoder
        import joblib
        joblib.dump(self.label_encoder, os.path.join(directory, "label_encoder.joblib"))
        logger.info(f"Model and tokenizer saved to {directory}")

    def load(self, directory: str):
        """
        Loads the model, tokenizer, and label encoder from disk.
        """
        if os.path.exists(directory):
            self.model = AutoModelForSequenceClassification.from_pretrained(directory).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(directory)
            
            import joblib
            le_path = os.path.join(directory, "label_encoder.joblib")
            if os.path.exists(le_path):
                self.label_encoder = joblib.load(le_path)
            logger.info(f"Model loaded from {directory}")
        else:
            raise DirectoryNotFoundError(f"Model directory {directory} not found.")

class DirectoryNotFoundError(Exception):
    pass
