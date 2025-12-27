import torch
import torch.nn as nn

class ConsistencyLoss(nn.Module):
    """
    Implements Eq. 13: L_total = L_BCE + lambda * L_consistency
    L_consistency ensures attention weights sum to 1 (soft constraint handled by softmax),
    but here we enforce distribution sparsity or specific consistency properties.
    
    As per paper text: "constrains the sum of the attention weights to equal 1... 
    The softmax already ensures sum=1, but the paper implies a regularization 
    to prevent trivial solutions or enforce competition."
    
    Given Eq 13: || sum(alpha) - 1 ||^2. Since Softmax output sums to 1 automatically, 
    this term might be intended to enforce sparsity or applied before softmax?
    
    However, following the specific text: "force sum... to approach 1". 
    If softmax is used, this loss is 0. 
    Likely interpretation: The authors might apply this on the *unnormalized* logits 
    OR they want to penalize deviation from a sharp distribution (entropy min).
    
    Based on the paper's "Explicit competition" description, we assume the regularization
    targets the stability of the distribution across the batch or enforces 
    that weights don't collapse to uniform [0.33, 0.33, 0.33].
    
    For strict adherence to the provided code snippet/formula in the manuscript,
    we implement the sum constraint check, but practically, we return 0 if softmax is used.
    """
    def __init__(self, lambda_reg=0.1):
        super(ConsistencyLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_reg = lambda_reg

    def forward(self, logits, targets, alphas):
        # 1. Classification Loss (BCE)
        loss_cls = self.bce(logits, targets)
        
        # 2. Consistency Regularization
        # Enforces the sum of attention weights to equal 1 (Eq. 13 in paper)
        # Note: Since we use softmax, sum is 1 by definition. 
        # This term is kept to strictly align with the mathematical formulation 
        # provided in the manuscript method section.
        sum_alphas = torch.sum(alphas, dim=1)
        loss_con = torch.mean((sum_alphas - 1.0) ** 2)
        
        total_loss = loss_cls + self.lambda_reg * loss_con
        return total_loss, loss_cls, loss_con