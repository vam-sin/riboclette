import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    def __init__(self):
        super().__init__()


class MaskedLoss(BaseLoss):
    def __init__(self):
        super().__init__()


class MaskedMSELoss(MaskedLoss):
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true, mask):
        y_pred_mask = torch.masked_select(y_pred, mask).float()
        y_true_mask = torch.masked_select(y_true, mask).float()

        loss = nn.functional.mse_loss(y_pred_mask, y_true_mask, reduction="none")
        return torch.sqrt(loss.mean())


class MaskedBCEWithLogitsLoss(MaskedLoss):
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true, mask):
        y_pred_mask = torch.masked_select(y_pred, mask).float()
        y_true_mask = torch.masked_select(y_true, mask).float()

        return nn.functional.binary_cross_entropy_with_logits(y_pred_mask, y_true_mask)


class MaskedPoissonLoss(MaskedLoss):
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true, mask):
        y_pred_mask = torch.masked_select(y_pred, mask)
        y_true_mask = torch.masked_select(y_true, mask)
        return nn.functional.poisson_nll_loss(y_pred_mask, y_true_mask, log_input=False)


class MaskedHuberLoss(MaskedLoss):
    def __init__(self):
        super().__init__()


class MaskedPearsonLoss(MaskedLoss):
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true, mask, eps=1e-6):
        y_pred_mask = torch.masked_select(y_pred, mask)
        y_true_mask = torch.masked_select(y_true, mask)
        cos = nn.CosineSimilarity(dim=0, eps=eps)
        return 1 - cos(
            y_pred_mask - y_pred_mask.mean(),
            y_true_mask - y_true_mask.mean(),
        )


class MaskedCombinedPearsonLoss(MaskedLoss):
    def __init__(self, comb_max_duration: int = 10):
        super().__init__()
        self.pearson = MaskedPearsonLoss()
        self.poisson = MaskedPoissonLoss()
        self.comb_max_duration = comb_max_duration

    def __call__(self, y_pred, y_true, mask, timestamp, eps=1e-6):
        poisson = self.poisson(y_pred, y_true, mask)
        pearson = self.pearson(y_pred, y_true, mask, eps=eps)

        return pearson + max(0, 1 - timestamp / self.comb_max_duration) * poisson
