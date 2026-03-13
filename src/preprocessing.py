import re
import string
import logging
from typing import List, Optional
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)

class SentimentPreprocessor:
    """
    Production-grade text preprocessing for Transformer-based Sentiment Analysis.
    Focuses on minimal cleaning that preserves context for models like RoBERTa.
    """
    def __init__(self):
        logger.info("SentimentPreprocessor initialized for Transformer models.")

    def clean_text(self, text: str) -> str:
        """
        Cleans raw tweet text from URLs, mentions, and extra whitespace.
        Preserves punctuation and casing as they provide context for Transformers.
        """
        if not isinstance(text, str):
            return ""

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user @mentions
        text = re.sub(r'\@\w+', '', text)
        
        # Replace #hashtags with just the word (optional, but often helpful)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def preprocess(self, text: str) -> str:
        """
        Standard preprocessing for Transformer models.
        """
        return self.clean_text(text)

    def preprocess_series(self, series: List[str]) -> List[str]:
        """Applies preprocessing to a list/series of strings."""
        logger.info(f"Preprocessing {len(series)} documents...")
        return [self.preprocess(text) for text in tqdm(series, desc="Cleaning text")]
