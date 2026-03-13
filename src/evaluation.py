import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Evaluates the performance of the sentiment model.
    """
    @staticmethod
    def evaluate(y_true, y_pred, labels=None, output_dir: str = "logs"):
        """
        Prints and plots evaluation metrics.
        """
        logger.info("Evaluating model performance...")
        
        # Accuracy
        acc = accuracy_score(y_true, y_pred)
        logger.info(f"Accuracy: {acc:.4f}")
        
        # Classification Report
        report = classification_report(y_true, y_pred, target_names=labels)
        print("\nClassification Report:\n", report)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plotting
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        logger.info(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")
        
        return {
            "accuracy": acc,
            "report": report,
            "confusion_matrix": cm
        }
