import torch
import numpy as np
from sklearn.metrics import roc_auc_score, mean_absolute_error

class MetricCalculator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []

    def update(self, targets, logits):
        """
        targets: (Batch, N)
        logits: (Batch, N) raw scores
        """
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        self.y_true.append(targets)
        self.y_pred.append(probs)

    def compute(self):
        y_true = np.concatenate(self.y_true, axis=0)
        y_pred = np.concatenate(self.y_pred, axis=0)
        
        # MAE
        mae = mean_absolute_error(y_true, y_pred)
        
        # Aggregate AUC (Eq. 14)
        try:
            auc = roc_auc_score(y_true, y_pred, average='macro')
        except ValueError:
            auc = 0.0
            
        return {'AUC': auc, 'MAE': mae}