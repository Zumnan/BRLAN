import torch
import torch.nn.functional as F

def tversky_bce_loss(predictions, ground_truths, smooth=1e-8, alpha=0.5, beta=0.5, pos_weight=1.0):
    """
    Combined Tversky Loss and Weighted Binary Cross-Entropy Loss for binary segmentation.
    
    Arguments:
    - predictions: The predicted logits from the model (before sigmoid activation).
    - ground_truths: The ground truth binary labels.
    - smooth: A smoothing factor to avoid division by zero (default 1e-8).
    - alpha: Weight for false positives in Tversky loss (default 0.5).
    - beta: Weight for false negatives in Tversky loss (default 0.5).
    - pos_weight: Weight for the positive class in weighted BCE (default 1.0).
    
    Returns:
    - Combined Tversky Loss and Weighted Binary Cross-Entropy Loss.
    """

    
    predictions = predictions.view(-1)
    ground_truths = ground_truths.view(-1).float()

    
    predictions_sigmoid = torch.sigmoid(predictions)

    # Calculate Tversky loss
    TP = (predictions_sigmoid * ground_truths).sum()
    FP = ((1 - ground_truths) * predictions_sigmoid).sum()
    FN = (ground_truths * (1 - predictions_sigmoid)).sum()

    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    tversky_loss = 1 - tversky

    # Calculate weighted binary cross-entropy loss
    weights = ground_truths * pos_weight + (1 - ground_truths)
    bce_loss = F.binary_cross_entropy_with_logits(predictions, ground_truths, weight=weights)

    
    return tversky_loss + bce_loss