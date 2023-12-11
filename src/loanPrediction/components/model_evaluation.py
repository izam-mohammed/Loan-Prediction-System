import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix
from loanPrediction.utils.common import load_bin, save_json, round_batch
from loanPrediction import logger
import matplotlib.pyplot as plt
import seaborn as sns
from loanPrediction.entity.config_entity import ModelEvaluationConfig
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
    

    def _eval_metrics(self,actual, pred):
        acc = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        rocauc_score = roc_auc_score(actual, pred)

        return round_batch(acc, precision, recall, f1, rocauc_score)
    
    def _eval_pics(self, actual, pred_proba):
        logger.info(actual.shape)
        fpr, tpr, thresholds = roc_curve(actual, pred_proba)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')

        plt.savefig(os.path.join(self.config.root_dir, "roc_curve.png"))
        plt.close()
        logger.info(f"Saved the ROC curve image at {os.path.join(self.config.root_dir, 'roc_curve.png')}")

        #confusion matrix
        y_pred = (pred_proba > 0.5).astype(int)
        cm = confusion_matrix(actual, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16})
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')

        plt.savefig(os.path.join(self.config.root_dir, "confusion_matrix.png"))
        plt.close()
        logger.info(f"Saved the cunfusion matrix image at {os.path.join(self.config.root_dir, 'confusion_matrix.png')}")


    def evaluate(self):
        df = pd.read_csv(self.config.test_data_path)
        X = df.drop([self.config.target_column], axis=1)
        y = df[self.config.target_column]

        model = load_bin(path=Path(self.config.model_path))
        pred = model.predict(X)
        pred_proba = model.predict_proba(X)[:, 1]

        logger.info(f"predicted {pred.shape[0]} data points")
    
        acc, precision, recall, f1, rocauc_score = self._eval_metrics(y, pred)
        self._eval_pics(y, pred_proba)

        metric = {
            "Accuracy" : acc,
            "Precision" : precision,
            "Recall": recall,
            "F1 score": f1,
            "roc auc score": rocauc_score,
        }

        logger.info(f"metrics are - {metric}")

        save_json(path=Path(self.config.metric_file_name), data=metric)