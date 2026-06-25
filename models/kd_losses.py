import torch
import torch.nn as nn
import torch.nn.functional as F

def kd_loss(logits_s, logits_t, temperature=4.0):
    p_s = F.log_softmax(logits_s / temperature, dim=1)
    p_t = F.softmax(logits_t / temperature, dim=1)
    return F.kl_div(p_s, p_t, reduction='batchmean') * (temperature ** 2)

def fitnet_loss(f_s_proj, f_t):
    return F.mse_loss(f_s_proj, f_t.detach())

def _single_at(feat, p=2):
    am = feat.pow(p).mean(dim=1)          # [B, H, W]
    am = am.view(am.size(0), -1)          # [B, H*W]
    am = F.normalize(am, p=2, dim=1)
    return am


def at_loss(feat_s, feat_t, p=2):
    assert feat_s.shape[2:] == feat_t.shape[2:], (
        f"AT requires matching spatial size, got student {feat_s.shape} "
        f"vs teacher {feat_t.shape}"
    )
    return (_single_at(feat_s, p) - _single_at(feat_t, p)).pow(2).mean()


def multi_scale_at_loss(feats_s, feats_t, p=2, weights=None):
    assert len(feats_s) == len(feats_t), "student/teacher feature list length mismatch"
    losses = [at_loss(fs, ft, p) for fs, ft in zip(feats_s, feats_t)]
    if weights is None:
        return sum(losses)
    assert len(weights) == len(losses)
    return sum(w * l for w, l in zip(weights, losses))

class SimKDProjector(nn.Module):
    def __init__(self, s_dim: int = 512, t_dim: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(s_dim, t_dim),
            nn.ReLU(inplace=True),
            nn.Linear(t_dim, t_dim),
        )

    def forward(self, f_s: torch.Tensor) -> torch.Tensor:
        return self.proj(f_s)


def simkd_loss(f_s_proj, f_t, logits_via_teacher_head, target, lam=1.0):
    ce_loss = F.cross_entropy(logits_via_teacher_head, target)
    mse_loss = F.mse_loss(f_s_proj, f_t.detach())
    return ce_loss + lam * mse_loss, ce_loss, mse_loss
