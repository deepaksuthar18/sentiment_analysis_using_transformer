import logging
import sys
import os
from src.pipeline import SentimentPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)

logger = logging.getLogger("Main")

def ensure_directories(config_path):
    """Ensures that all paths required by the config exist."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    paths = config.get('paths', {})
    for path_name, path_val in paths.items():
        if path_name.endswith('_dir') or path_name.endswith('_path'):
            if not os.path.exists(path_val):
                os.makedirs(path_val)
                logger.info(f"Created directory: {path_val}")

def main():
    try:
        config_path = "config/config_processed.yaml"
        logger.info("Initializing RoBERTa Sentiment Analysis Pipeline...")
        
        # Ensure directories exist
        ensure_directories(config_path)
        
        # Initialize the pipeline
        pipeline = SentimentPipeline(config_path=config_path)
        
        # Run training and evaluation
        results = pipeline.run_training_pipeline()
        
        logger.info(f"Training Complete. Validation Accuracy: {results['accuracy']:.4f}")
        logger.info("Project executed successfully.")

    except Exception as e:
        logger.error(f"Critical error in pipeline: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
