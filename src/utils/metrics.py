import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import pearson_corrcoef
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_f1_score,
    binary_precision,
    binary_recall,
)


class CorrCoef(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("corrcoefs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, mask: Tensor):
        assert preds.shape == target.shape
        assert preds.shape == mask.shape

        coeffs = []
        for p, t, m in zip(preds, target, mask):
            mp, mt = torch.masked_select(p, m), torch.masked_select(t, m)
            temp_pearson = pearson_corrcoef(mp, mt)

            coeffs.append(temp_pearson)

        coeffs = torch.stack(coeffs)
        self.corrcoefs += torch.sum(coeffs)
        self.total += len(coeffs)

    def compute(self):
        return self.corrcoefs / self.total


class BinaryMaskedMetric(Metric):
    def __init__(self, metric_name: str):
        super().__init__()
        self.add_state("vals", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        if metric_name == "accuracy":
            self.metric = binary_accuracy
        elif metric_name == "precision":
            self.metric = binary_precision
        elif metric_name == "recall":
            self.metric = binary_recall
        elif metric_name == "f1_score":
            self.metric = binary_f1_score
        else:
            raise ValueError()

    def update(self, preds: Tensor, target: Tensor, mask: Tensor):
        assert preds.shape == target.shape
        assert preds.shape == mask.shape

        vals = []
        for p, t, m in zip(preds, target, mask):
            mp, mt = torch.masked_select(p, m), torch.masked_select(t, m)
            acc = self.metric(mp, mt)

            vals.append(acc)

        self.vals += torch.stack(vals).sum()
        self.total += len(vals)

    def compute(self):
        return self.vals / self.total
