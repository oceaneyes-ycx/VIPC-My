"""Per-sample metrics using the same CUDA Chamfer distances as training."""
import torch
from extensions.chamfer_dist import ChamferFunction


def completion_metrics(prediction, target, threshold=0.001):
    """CD-L1 is half the directional sum; CD-L2 is the full sum.

    F-score uses a threshold on SQUARED Euclidean distances, as in the
    historical ViPC evaluation. CD values are returned without x1000 scaling.
    """
    d1, d2 = ChamferFunction.apply(prediction.contiguous(), target.contiguous())
    cd_l1 = (d1.sqrt().mean(1) + d2.sqrt().mean(1)) / 2
    cd_l2 = d1.mean(1) + d2.mean(1)
    precision = (d1 < threshold).float().mean(1)
    recall = (d2 < threshold).float().mean(1)
    fscore = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return torch.stack((cd_l1, cd_l2, fscore), dim=1)
