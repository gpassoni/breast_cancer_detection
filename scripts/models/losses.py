import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(1, 2, 3))
        den = (probs + targets).sum(dim=(1, 2, 3)) + self.smooth
        dice = num / den
        return 1 - dice.mean()

class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return self.alpha * self.bce(logits, targets) + (1 - self.alpha) * self.dice(
            logits, targets
        )


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.clamp(min=1e-6, max=1 - 1e-6)  
        targets = targets.float()

        loss = -self.alpha * (
            targets * torch.pow(1 - probs, self.gamma) * torch.log(probs)
        ) - (1 - self.alpha) * (
            (1 - targets) * torch.pow(probs, self.gamma) * torch.log(1 - probs)
        )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalLossMulticlass(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        log_probs = F.log_softmax(logits, dim=1)           
        probs = torch.exp(log_probs)                       

        targets_onehot = F.one_hot(targets, num_classes)   
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  

        pt = (probs * targets_onehot).sum(dim=1)           
        log_pt = (log_probs * targets_onehot).sum(dim=1)   

        focal_term = (1 - pt).clamp(min=1e-4) ** self.gamma
        loss = - focal_term * log_pt  

        if self.alpha is not None:
            if self.alpha.type() != logits.data.type():
                self.alpha = self.alpha.type_as(logits.data)

            at = self.alpha[targets]  
            loss *= at

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  




class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(logits.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        TP = (probs * targets).sum(dim=1)
        FP = (probs * (1 - targets)).sum(dim=1)
        FN = ((1 - probs) * targets).sum(dim=1)

        tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )
        return 1 - tversky.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 1e-4, 1.0 - 1e-4)  
        probs = probs.view(logits.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        TP = (probs * targets).sum(dim=1)
        FP = (probs * (1 - targets)).sum(dim=1)
        FN = ((1 - probs) * targets).sum(dim=1)

        tversky_index = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )
        tversky_index = torch.clamp(tversky_index, 1e-4, 1.0 - 1e-4)
        focal_tversky = torch.pow((1 - tversky_index), self.gamma)
        return focal_tversky.mean()
