import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve


def tpr_at_fpr(y, s, target_fpr=0.05):
    """Highest true-positive rate achievable with false-positive rate <= target."""
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def all_metrics(y, s, threshold=0.5):
    y, s = np.asarray(y), np.asarray(s)
    pred = (s >= threshold).astype(int)
    return dict(auroc=float(roc_auc_score(y, s)),
                tpr_at_5fpr=tpr_at_fpr(y, s, 0.05),
                acc=float(accuracy_score(y, pred)),
                f1=float(f1_score(y, pred)))
