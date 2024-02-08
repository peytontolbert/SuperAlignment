import torch.nn as nn
import torch.nn.functional as F


# Custom Adaptive Confidence Distillation Loss
class AdaptiveConfidenceDistillationLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        # Soften probabilities
        soft_teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student_probs = F.log_softmax(student_logits / self.temperature, dim=1)

        # Calculate the KL Divergence for the soft targets
        distillation_loss = F.kl_div(
            soft_student_probs, soft_teacher_probs, reduction="batchmean"
        )

        # Calculate the standard loss with hard targets
        hard_loss = self.ce_loss(student_logits, targets)

        # Adaptive weighting could be implemented here based on confidence
        # For simplicity, this example uses a fixed ratio
        return distillation_loss + hard_loss
