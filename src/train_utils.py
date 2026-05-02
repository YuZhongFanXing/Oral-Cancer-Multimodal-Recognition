"""Training utilities: EMA, loss function, evaluation, and reproducibility."""

import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, matthews_corrcoef


def set_seed(seed):
    """Reproducibility: set all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ModelEMA:
    """Exponential Moving Average for model parameters.

    Maintains shadow weights updated as:
      shadow = decay * shadow + (1 - decay) * current_params
    """
    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters()}

    def update(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                self.shadow[n] = (self.decay * self.shadow[n]
                                  + (1 - self.decay) * p.data)

    def apply_shadow(self, model):
        self._backup = {n: p.data.clone() for n, p in model.named_parameters()}
        for n, p in model.named_parameters():
            p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            p.data.copy_(self._backup[n])

    class _Ctx:
        def __init__(self, ema, model):
            self.ema, self.model = ema, model

        def __enter__(self):
            self.ema.apply_shadow(self.model)
            return self.model

        def __exit__(self, *a):
            self.ema.restore(self.model)

    def get_context(self, model):
        return self._Ctx(self, model)


class WeightedLabelSmoothingLoss(nn.Module):
    """Cross-entropy with label smoothing and class weights.

    smoothing=0.05: target distribution becomes [0.025, 0.975] for class 1
    weight=[2, 1]: benign errors cost 2x more than malignant errors
    """
    def __init__(self, classes=2, smoothing=0.05, dim=-1, weight=None):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.weight = weight

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        if self.weight is not None:
            true_dist = true_dist * self.weight.to(pred.device).view(1, -1)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def find_threshold(labels, probs, thr_range=None):
    """Search for the optimal threshold maximizing balanced accuracy."""
    if thr_range is None:
        thr_range = np.arange(0.15, 0.95, 0.01)
    best_score, best_th = 0.0, 0.5
    labels = np.array(labels)
    for th in thr_range:
        preds = (probs[:, 1] >= th).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        bal = (tp / (tp + fn + 1e-6) + tn / (tn + fp + 1e-6)) / 2.0
        if bal > best_score:
            best_score, best_th = bal, th
    return best_th


def evaluate(model, loader, criterion, device, threshold=0.5, tta=False):
    """Evaluate model on a data loader.

    TTA: 3-way flip averaging (original, h-flip, v-flip).
    """
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0, [], [], []
    with torch.no_grad():
        for img, meta, lbl in loader:
            img, meta, lbl = img.to(device), meta.to(device), lbl.to(device)
            if tta:
                out = (model(img, meta)
                       + model(torch.flip(img, [3]), meta)
                       + model(torch.flip(img, [2]), meta)) / 3.0
            else:
                out = model(img, meta)
            if criterion is not None:
                total_loss += criterion(out, lbl).item() * len(lbl)
            pb = torch.softmax(out, dim=1)
            all_probs.extend(pb.cpu().numpy())
            all_preds.extend((pb[:, 1] >= threshold).long().cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())

    loss = total_loss / len(loader.dataset) if criterion is not None else 0
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return acc, loss, np.array(all_preds), np.array(all_labels), np.array(all_probs)
