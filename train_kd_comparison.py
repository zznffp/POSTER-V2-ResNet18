import warnings
warnings.filterwarnings("ignore")
import os
import sys
import argparse
import datetime
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from thop import profile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
from models.PosterV2_7cls import PosterV2_ResNet
from models.PosterV2_Original import pyramid_trans_expr2
from models.kd_losses import kd_loss, fitnet_loss, multi_scale_at_loss, SimKDProjector, simkd_loss

# NA-MSAC module (our method, only used for --kd_method ours)
try:
    from noise_aware_native_attention import NoiseAwareNativeAttention

    NA_MSAC_AVAILABLE = True
except ImportError:
    print("Warning: NA-MSAC module not found. --lambda_na_msac will be disabled.")
    NA_MSAC_AVAILABLE = False

def compute_kd_baseline_loss(method, args,
                             student_output, student_features,
                             teacher_output, teacher_features,
                             images, target, student, teacher,
                             criterion_hard, feature_projector, simkd_projector):
    device = student_output.device
    hard_loss = criterion_hard(student_output, target)
    soft_loss = torch.zeros((), device=device)
    feature_loss = torch.zeros((), device=device)

    # ---- KD [Hinton 2015]: softened-logit KL -----------------------------
    if method == 'kd':
        soft_loss = kd_loss(student_output, teacher_output, args.temperature)
        dist_loss = (1.0 - args.alpha) * hard_loss + args.alpha * soft_loss

    # ---- FitNet [Romero 2015]: raw intermediate-feature MSE (no KL) -------
    elif method == 'fitnet':
        s_cls = student_features[:, 0, :]                  # [B, 512]
        t_cls = teacher_features[:, 0, :]                  # [B, 768]
        s_cls_proj = feature_projector(s_cls)              # 512 -> 768
        feature_loss = fitnet_loss(s_cls_proj, t_cls)      # plain MSE, no L2 norm
        dist_loss = hard_loss + args.fitnet_lambda * feature_loss

    # ---- AT [Zagoruyko 2017]: spatial attention transfer, NO KL term -----
    elif method == 'at':
        backbone = student.module if hasattr(student, 'module') else student
        s_feats = backbone.forward_backbone_adapted(images)        # 3 scales, student
        with torch.no_grad():
            t_feats = teacher(images, return_backbone_feats=True)  # 3 scales, teacher
        at_val = multi_scale_at_loss(list(s_feats), list(t_feats), p=2)  # summed over scales
        dist_loss = hard_loss + args.at_beta * at_val
        feature_loss = at_val.detach()

    # ---- SimKD [Chen 2022]: projector + frozen teacher-head reuse --------
    elif method == 'simkd':
        s_cls = student_features[:, 0, :]                  # [B, 512]
        t_cls_pre = teacher_features[:, 0, :]              # [B, 768] pre-SE
        with torch.no_grad():
            t_cls = teacher.VIT.se_block(t_cls_pre)        # post-SE target for MSE
        f_s_proj = simkd_projector(s_cls)                  # [B, 768]
        f_s_proj_se = teacher.VIT.se_block(f_s_proj)       # match teacher-head input dist
        logits_via_teacher = teacher.VIT.head(f_s_proj_se)  # reuse frozen head
        simkd_total, ce_via_teacher, mse_val = simkd_loss(
            f_s_proj, t_cls, logits_via_teacher, target, lam=args.simkd_lambda
        )
        # keep hard_loss so the student's own head is still trained (validate() uses it)
        dist_loss = hard_loss + simkd_total
        soft_loss = ce_via_teacher
        feature_loss = mse_val.detach()

    else:
        raise ValueError(f"compute_kd_baseline_loss got unsupported method: {method}")

    return dist_loss, hard_loss, soft_loss, feature_loss

class FocalLoss(nn.Module):
    """Focal Loss for class imbalance / hard samples. Used as the hard-label
    loss for every KD method to keep row-wise fairness on imbalanced FER data."""

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, use_distance=True):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.use_distance = use_distance

    def forward(self, student_features, teacher_features):
        batch_size = student_features.shape[0]
        student_features = F.normalize(student_features, p=2, dim=1)
        teacher_features = F.normalize(teacher_features, p=2, dim=1)

        if self.use_distance:
            D_s = torch.cdist(student_features, student_features, p=2)
            D_t = torch.cdist(teacher_features, teacher_features, p=2)
            D_s = D_s - torch.min(D_s)
            D_t = D_t - torch.min(D_t)
            S_s = torch.max(D_s) - D_s
            S_t = torch.max(D_t) - D_t
            S_s = F.normalize(S_s, p=2, dim=1)
            S_t = F.normalize(S_t, p=2, dim=1)
            return F.mse_loss(S_s, S_t)
        else:
            similarity_matrix = torch.matmul(student_features, teacher_features.T) / self.temperature
            labels = torch.arange(batch_size).to(student_features.device)
            loss_s2t = F.cross_entropy(similarity_matrix, labels)
            loss_t2s = F.cross_entropy(similarity_matrix.T, labels)
            return (loss_s2t + loss_t2s) / 2.0

def mixup_data(x, y, alpha=0.2, device='cuda'):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str,
                    default='./data_preprocessing/val_datasets/raf-db-divide-7folder',
                    help='dataset path')
parser.add_argument('--teacher_path', type=str,
                    default='./models/pretrain/raf-db-model_best.pth',
                    help='Teacher model checkpoint path')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/resnet_distill_model.pth')
parser.add_argument('--best_checkpoint_path', type=str, default='./checkpoints/resnet_distill_model_best.pth')
parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
parser.add_argument('--start_epoch', type=int, default=0, help='Manual start epoch (use with --resume)')
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batch-size', default=64, type=int, help='Batch Size')
parser.add_argument('--lr', default=0.0001, type=float, help='Learning Rate')
parser.add_argument('--beta', default=0.5, type=float, help='Aux Loss Weight (architectural constant)')
parser.add_argument('--alpha', default=0.6, type=float, help='Hinton KD weight (used by kd / ours)')
parser.add_argument('--temperature', default=4.0, type=float, help='Distillation Temperature')
parser.add_argument('--dropout', default=0.4, type=float, help='Dropout rate')
parser.add_argument('--lambda_feature', default=0.7, type=float, help='Feature KD weight (ours only)')
parser.add_argument('--lambda_contrast', default=0.0, type=float, help='Contrastive KD weight (ours only)')
parser.add_argument('--lambda_proto', default=0.0, type=float, help='Prototype KD weight (ours only)')
parser.add_argument('--contrast_temperature', default=0.07, type=float, help='Contrastive temperature')
parser.add_argument('--use_distance_contrast', default=True, type=bool, help='Distance-based contrastive (QCS)')
parser.add_argument('--warmup_epochs', default=5, type=int, help='Warmup epochs')
parser.add_argument('--accumulation_steps', default=1, type=int, help='Gradient accumulation steps')
parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay')
parser.add_argument('--use_mixup', default=True, type=bool, help='Use Mixup (ours only; forced off for baselines)')
parser.add_argument('--mixup_alpha', default=0.2, type=float, help='Mixup alpha')
parser.add_argument('--mixup_prob', default=0.5, type=float, help='Probability of applying Mixup')
parser.add_argument('--seed', type=int, default=None, help='Random seed (None for time-based)')
parser.add_argument('--focal_gamma', default=2.0, type=float, help='Focal Loss gamma')
parser.add_argument('--lambda_na_msac', type=float, default=1.0, help='NA-MSAC loss weight (ours only)')
parser.add_argument('--na_msac_noise_aware', action='store_true', default=True, help='NA-MSAC noise-aware weighting')
parser.add_argument('--na_msac_noise_threshold', type=float, default=0.3, help='NA-MSAC noise threshold')
parser.add_argument('--na_msac_class_aware', action='store_true', default=False, help='NA-MSAC class-aware weighting')
parser.add_argument('--no_na_msac_class_aware', action='store_false', dest='na_msac_class_aware',
                    help='Disable NA-MSAC class-aware weighting')
parser.add_argument('--use_csi', action='store_true', default=False, help='Enable Cross-Scale Interaction')
parser.add_argument('--no_csi', action='store_false', dest='use_csi', help='Disable Cross-Scale Interaction')

# ===== KD comparison switch (tab:kd_comparison) =====
parser.add_argument('--kd_method', type=str, default='ours',
                    choices=['ours', 'kd', 'fitnet', 'at', 'simkd'],
                    help='KD baseline selector for the comparison experiment')
parser.add_argument('--at_beta', type=float, default=1000.0, help='AT loss weight (mdistiller default 1000)')
parser.add_argument('--simkd_lambda', type=float, default=1.0, help='SimKD MSE weight (paper default 1.0)')
parser.add_argument('--fitnet_lambda', type=float, default=0.7,
                    help='FitNet MSE weight (matched to lambda_feature for fairness)')

args = parser.parse_args()


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class RecorderMeter(object):
    def __init__(self, total_epoch):
        self.total_epoch = total_epoch
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss
        self.epoch_losses[idx, 1] = val_loss
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc

    def plot_curve(self, save_path):
        plt.figure(figsize=(12, 6))
        x = np.arange(self.total_epoch)
        plt.plot(x, self.epoch_accuracy[:, 0], label='Train Acc')
        plt.plot(x, self.epoch_accuracy[:, 1], label='Val Acc')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.close()


def multilayer_distillation_loss(student_logits, teacher_logits,
                                 student_features, teacher_features,
                                 labels, temperature, alpha,
                                 lambda_feature, lambda_contrast, criterion_hard, contrastive_criterion,
                                 feature_projector=None):
    hard_loss = criterion_hard(student_logits, labels)

    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)

    student_cls = student_features[:, 0, :]
    teacher_cls = teacher_features[:, 0, :]
    if feature_projector is not None:
        student_cls = feature_projector(student_cls)
    student_cls_norm = F.normalize(student_cls, p=2, dim=-1)
    teacher_cls_norm = F.normalize(teacher_cls, p=2, dim=-1)
    cls_loss = F.mse_loss(student_cls_norm, teacher_cls_norm)

    student_global = student_features[:, 1:, :].mean(dim=1)
    teacher_global = teacher_features[:, 1:, :].mean(dim=1)
    if feature_projector is not None:
        student_global = feature_projector(student_global)
    student_global_norm = F.normalize(student_global, p=2, dim=-1)
    teacher_global_norm = F.normalize(teacher_global, p=2, dim=-1)
    global_loss = F.mse_loss(student_global_norm, teacher_global_norm)

    feature_loss = 0.6 * cls_loss + 0.4 * global_loss
    contrastive_loss = contrastive_criterion(student_cls, teacher_cls)

    logits_loss = alpha * soft_loss + (1 - alpha) * hard_loss
    total_loss = logits_loss + lambda_feature * feature_loss + lambda_contrast * contrastive_loss
    return total_loss, hard_loss, soft_loss, feature_loss, contrastive_loss


def load_teacher_model(checkpoint_path):
    import sys
    original_recorder = sys.modules['__main__'].__dict__.get('RecorderMeter', None)
    original_recorder1 = sys.modules['__main__'].__dict__.get('RecorderMeter1', None)

    class TempRecorderMeter:
        pass

    class TempRecorderMeter1:
        pass

    sys.modules['__main__'].RecorderMeter = TempRecorderMeter
    sys.modules['__main__'].RecorderMeter1 = TempRecorderMeter1

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if original_recorder is not None:
        sys.modules['__main__'].RecorderMeter = original_recorder
    if original_recorder1 is not None:
        sys.modules['__main__'].RecorderMeter1 = original_recorder1

    teacher = pyramid_trans_expr2(img_size=224, num_classes=7)
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k[7:] if k.startswith('module.') else k] = v
    teacher.load_state_dict(new_state_dict, strict=False)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print(f"   Teacher accuracy: {checkpoint['best_acc']:.2f}%")
    return teacher


def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    if not os.path.exists('./log'):
        os.makedirs('./log')

    args = parser.parse_args()

    if args.kd_method != 'ours':
        if args.lambda_na_msac > 0:
            print(f"[{args.kd_method}] force NA-MSAC OFF (lambda_na_msac={args.lambda_na_msac} -> 0)")
        args.lambda_na_msac = 0.0
        if args.use_mixup:
            print(f"[{args.kd_method}] force Mixup OFF (avoids target-mask ambiguity)")
        args.use_mixup = False
        if args.lambda_feature > 0:
            print(f"[{args.kd_method}] force Feature KD OFF (lambda_feature={args.lambda_feature} -> 0)")
        args.lambda_feature = 0.0
        if args.lambda_contrast > 0:
            print(f"[{args.kd_method}] force Contrast KD OFF (lambda_contrast={args.lambda_contrast} -> 0)")
        args.lambda_contrast = 0.0

    if args.seed is not None:
        seed = args.seed
        seed_type = "user-specified"
        print(f"Using user-specified seed: {seed}")
    else:
        seed = int(time.time() * 1000) % (2 ** 32)
        if len(str(seed)) % 2 != 0:
            seed = seed // 10
        seed_type = "auto"
        print(f"Using auto-generated seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducibility")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f'./log/train_{args.kd_method}_seed{seed}_{timestamp}.log'
    curve_file = f'./log/train_{args.kd_method}_seed{seed}_{timestamp}_curve.png'
    args.best_checkpoint_path = f'./checkpoints/resnet_distill_{args.kd_method}_seed{seed}_best_{timestamp}.pth'

    def log_print(message):
        print(message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

    log_print("=" * 80)
    log_print("KD Comparison Training (tab:kd_comparison)")
    log_print("=" * 80)
    teacher = load_teacher_model(args.teacher_path)
    teacher = teacher.cuda()

    log_print("\n=> Creating student model: PosterV2 (ResNet18)...")
    student = PosterV2_ResNet(img_size=224, num_classes=7, dropout=args.dropout, use_csi=args.use_csi)

    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    input_tensor = torch.randn(1, 3, 224, 224)
    flops, _ = profile(student, inputs=(input_tensor,), verbose=False)
    flops_g = flops / 1e9

    log_print(
        f"   Student Parameters: {trainable_params / 1e6:.2f}M (trainable) + "
        f"{frozen_params / 1e6:.2f}M (frozen) = {total_params / 1e6:.2f}M (total)")
    log_print(f"   Student FLOPs: {flops_g:.2f} G")
    log_print(f"   Log file: {log_file}")
    log_print(f"   Training started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"\n   === General Training Settings ===")
    log_print(f"   - Random seed: {seed} ({seed_type})")
    log_print(f"   - Batch size: {args.batch_size}")
    log_print(f"   - Learning rate: {args.lr}")
    log_print(f"   - Weight decay: {args.weight_decay}")
    log_print(f"   - Epochs: {args.epochs}")
    log_print(f"   - Warmup epochs: {args.warmup_epochs}")
    log_print(f"   - LR Scheduler: Warmup + Cosine Annealing")
    log_print(f"   - Focal Loss gamma: {args.focal_gamma}")
    log_print(f"   - Dropout: {args.dropout}")
    log_print(f"   - Aux loss weight (beta): {args.beta}")

    # ---- KD method-specific banner ----
    log_print(f"\n   === KD Method: [{args.kd_method.upper()}] ===")
    if args.kd_method == 'ours':
        log_print(f"   Formula: L = (1-a)*hard + a*KL*T^2 + lf*feat + lc*contrast "
                  f"+ lp*proto + lna*NA-MSAC + b*aux")
        log_print(f"   - alpha (a): {args.alpha}   Temperature (T): {args.temperature}")
        log_print(f"   - lambda_feature (lf): {args.lambda_feature}")
        log_print(f"   - lambda_contrast (lc): {args.lambda_contrast} (distance={args.use_distance_contrast})")
        log_print(f"   - lambda_proto (lp): {args.lambda_proto}")
        log_print(f"   - lambda_na_msac (lna): {args.lambda_na_msac}")
        log_print(f"   - Mixup: {args.use_mixup} (alpha={args.mixup_alpha}, prob={args.mixup_prob})")
    elif args.kd_method == 'kd':
        log_print(f"   Formula (Hinton NeurIPS'15): L = (1-a)*hard + a*KL*T^2 + b*aux")
        log_print(f"   - alpha (a): {args.alpha}   Temperature (T): {args.temperature}")
        log_print(f"   - Feature KD / Contrast / NA-MSAC / Mixup: OFF (A-plan)")
    elif args.kd_method == 'fitnet':
        log_print(f"   Formula (Romero ICLR'15): L = hard + lf_fitnet*MSE(proj(f_s), f_t) + b*aux")
        log_print(f"   - feature alignment: raw MSE, NO L2 normalization (strict original)")
        log_print(f"   - feature: student CLS [B,512] -> proj -> [B,768] vs teacher CLS [B,768]")
        log_print(f"   - fitnet_lambda: {args.fitnet_lambda}")
        log_print(f"   - KL / Contrast / NA-MSAC / Mixup: OFF (A-plan)")
    elif args.kd_method == 'at':
        log_print(f"   Formula (Zagoruyko ICLR'17 / mdistiller): L = hard + b_at*AT + b*aux  (NO KL term)")
        log_print(f"   - AT: p=2, L2-normalize(mean_c|f|^2).flatten -> MSE, summed over 3 scales")
        log_print(f"   - scales: [B,64,28,28] / [B,128,14,14] / [B,256,7,7]")
        log_print(f"   - student: forward_backbone_adapted();  teacher: forward(return_backbone_feats=True)")
        log_print(f"   - at_beta: {args.at_beta}")
        log_print(f"   - KL / Feature KD / Contrast / NA-MSAC / Mixup: OFF (A-plan)")
    elif args.kd_method == 'simkd':
        log_print(f"   Formula (Chen CVPR'22): L = hard + CE(h_T(proj(f_s)), y) + l*MSE(proj(f_s), f_t) + b*aux")
        log_print(f"   - projector: Linear(512->768)+ReLU+Linear(768->768)")
        log_print(f"   - teacher head h_T: teacher.VIT.head (frozen, post-SE input)")
        log_print(f"   - simkd_lambda: {args.simkd_lambda}")
        log_print(f"   - KL / Feature KD / Contrast / NA-MSAC / Mixup: OFF (A-plan)")
    log_print("-" * 80)

    student = student.cuda()

    # feature projector (512 -> 768); reused by 'ours' and 'fitnet'
    feature_projector = nn.Linear(512, 768).cuda()
    log_print(f"   - Feature projector: 512 -> 768")

    # NA-MSAC (ours only)
    na_msac_module = None
    if args.lambda_na_msac > 0:
        if not NA_MSAC_AVAILABLE:
            log_print("   ERROR: NA-MSAC requested but module not available!")
            exit(1)
        na_msac_module = NoiseAwareNativeAttention(
            num_classes=7,
            feature_dims=[64, 128, 256],
            feature_sizes=[28, 14, 7],
            scale_weights=[0.2, 0.3, 0.5],
            use_noise_aware=args.na_msac_noise_aware,
            noise_threshold=args.na_msac_noise_threshold,
            use_class_aware=args.na_msac_class_aware
        ).cuda()
        log_print(f"   [NA-MSAC] Lambda: {args.lambda_na_msac}")
        log_print(f"   [NA-MSAC] Parameters: {sum(p.numel() for p in na_msac_module.parameters())}")

    # data
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random')
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(args.data, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(args.data, 'valid'), transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # losses / optimizer
    criterion_focal = FocalLoss(alpha=1.0, gamma=args.focal_gamma).cuda()
    criterion_contrastive = ContrastiveLoss(
        temperature=args.contrast_temperature, use_distance=args.use_distance_contrast).cuda()

    # SimKD projector (simkd only)
    simkd_projector = None
    if args.kd_method == 'simkd':
        simkd_projector = SimKDProjector(s_dim=512, t_dim=768).cuda()
        log_print(f"   [SimKD] Projector: Linear(512->768)+ReLU+Linear(768->768), "
                  f"{sum(p.numel() for p in simkd_projector.parameters()) / 1e3:.1f}K params")

    opt_params = list(student.parameters()) + list(feature_projector.parameters())
    if simkd_projector is not None:
        opt_params += list(simkd_projector.parameters())
    optimizer = optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = get_lr_scheduler(optimizer, args.warmup_epochs, args.epochs)

    recorder = RecorderMeter(args.epochs)
    best_acc = 0.0
    start_epoch = args.start_epoch

    if args.resume:
        if os.path.isfile(args.resume):
            log_print(f"\n=> Loading checkpoint from '{args.resume}'")
            checkpoint = torch.load(args.resume)
            if 'epoch' in checkpoint and args.start_epoch == 0:
                start_epoch = checkpoint['epoch'] + 1
                best_acc = checkpoint['best_acc']
                student.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                if 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
                if 'recorder' in checkpoint:
                    recorder.epoch_losses[:start_epoch] = checkpoint['recorder']['losses'][:start_epoch]
                    recorder.epoch_accuracy[:start_epoch] = checkpoint['recorder']['accuracy'][:start_epoch]
                log_print(f"   Resuming from epoch {start_epoch}")
            else:
                best_acc = checkpoint.get('best_acc', 0.0)
                student.load_state_dict(checkpoint['state_dict'])
                log_print(f"   Loaded model weights (start epoch {start_epoch})")
            log_print(f"   Best accuracy so far: {best_acc:.2f}%")
        else:
            log_print(f"=> No checkpoint found at '{args.resume}'")

    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()
        log_print(f"   Adjusted learning rate to: {optimizer.param_groups[0]['lr']:.6f} for epoch {start_epoch}")

    # training loop
    teacher_prototypes = {}
    for epoch in range(start_epoch, args.epochs):
        train_acc, train_loss = train_distill(
            train_loader, student, teacher, criterion_focal, criterion_contrastive,
            optimizer, epoch, scaler, args, log_file,
            feature_projector=feature_projector, na_msac_module=na_msac_module,
            teacher_prototypes=teacher_prototypes, simkd_projector=simkd_projector)

        val_acc, val_loss = validate(val_loader, student, criterion_focal)
        scheduler.step()
        recorder.update(epoch, train_loss, train_acc, val_loss, val_acc)
        recorder.plot_curve(curve_file)

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        current_lr = optimizer.param_groups[0]['lr']

        log_print(f"Epoch [{epoch}/{args.epochs}] LR: {current_lr:.6f} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Best Acc: {best_acc:.2f}%")

        if is_best:
            torch.save({
                'epoch': epoch, 'state_dict': student.state_dict(), 'best_acc': best_acc,
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(), 'seed': seed, 'seed_type': seed_type,
                'timestamp': timestamp, 'config': vars(args),
                'recorder': {'losses': recorder.epoch_losses, 'accuracy': recorder.epoch_accuracy}
            }, args.best_checkpoint_path)
            log_print(f"  Best model saved to {args.best_checkpoint_path}")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch, 'state_dict': student.state_dict(), 'best_acc': best_acc,
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(), 'seed': seed, 'seed_type': seed_type,
                'timestamp': timestamp, 'config': vars(args),
                'recorder': {'losses': recorder.epoch_losses, 'accuracy': recorder.epoch_accuracy}
            }, args.checkpoint_path)
            log_print(f"  Checkpoint saved to {args.checkpoint_path}")

    log_print("-" * 80)
    log_print(f"Training completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Best validation accuracy: {best_acc:.2f}%")


def train_distill(train_loader, student, teacher, criterion_hard, criterion_contrastive, optimizer, epoch, scaler, args,
                  log_file=None, feature_projector=None, na_msac_module=None,
                  teacher_prototypes=None, simkd_projector=None):
    losses = AverageMeter('Loss', ':.4f')
    hard_losses = AverageMeter('Hard', ':.4f')
    soft_losses = AverageMeter('Soft', ':.4f')
    feature_losses = AverageMeter('Feature', ':.4f')
    contrastive_losses = AverageMeter('Contrast', ':.4f')
    proto_losses = AverageMeter('Proto', ':.4f')
    na_msac_losses = AverageMeter('NA-MSAC', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')

    student.train()
    teacher.eval()
    is_ours = (args.kd_method == 'ours')

    for i, (images, target) in enumerate(train_loader):
        images, target = images.cuda(), target.cuda()

        if na_msac_module is not None:
            images_flip = torch.flip(images, dims=[3])

        use_mixup_this_batch = args.use_mixup and np.random.rand() < args.mixup_prob
        if use_mixup_this_batch:
            images, target_a, target_b, lam = mixup_data(images, target, alpha=args.mixup_alpha, device='cuda')

        with torch.cuda.amp.autocast():
            student_output, student_aux, _, student_features = student(images, return_features=True)

            # adapted features + native attention maps (ours / NA-MSAC only)
            if na_msac_module is not None:
                x1_adapted, x2_adapted, x3_adapted = student.forward_backbone_adapted(images)
                x1_adapted_flip, x2_adapted_flip, x3_adapted_flip = student.forward_backbone_adapted(images_flip)
                _, _, attn_maps_orig = student(images, return_attention=True)
                _, _, attn_maps_flip = student(images_flip, return_attention=True)

            with torch.no_grad():
                teacher_output, _, _, teacher_features = teacher(images, return_features=True)

            if not is_ours:
                # ----- KD baselines (kd / fitnet / at / simkd) -----
                dist_loss, hard_loss, soft_loss, feature_loss = compute_kd_baseline_loss(
                    args.kd_method, args, student_output, student_features,
                    teacher_output, teacher_features, images, target, student, teacher,
                    criterion_hard, feature_projector, simkd_projector)
                contrastive_loss = torch.zeros((), device=images.device)
                proto_loss = torch.zeros((), device=images.device)
                aux_loss = criterion_hard(student_aux, target)

            elif use_mixup_this_batch:
                # ----- ours, mixup batch -----
                hard_loss = mixup_criterion(criterion_hard, student_output, target_a, target_b, lam)
                soft_loss = F.kl_div(
                    F.log_softmax(student_output / args.temperature, dim=1),
                    F.softmax(teacher_output / args.temperature, dim=1),
                    reduction='batchmean') * (args.temperature ** 2)

                student_cls = student_features[:, 0, :]
                teacher_cls = teacher_features[:, 0, :]
                student_cls_projected = feature_projector(student_cls)
                cls_loss = F.mse_loss(F.normalize(student_cls_projected, p=2, dim=-1),
                                      F.normalize(teacher_cls, p=2, dim=-1))
                student_global = student_features[:, 1:, :].mean(dim=1)
                teacher_global = teacher_features[:, 1:, :].mean(dim=1)
                student_global_projected = feature_projector(student_global)
                global_loss = F.mse_loss(F.normalize(student_global_projected, p=2, dim=-1),
                                         F.normalize(teacher_global, p=2, dim=-1))
                feature_loss = 0.6 * cls_loss + 0.4 * global_loss
                contrastive_loss = criterion_contrastive(student_cls, teacher_cls)
                aux_loss = mixup_criterion(criterion_hard, student_aux, target_a, target_b, lam)
                proto_loss = torch.zeros((), device=images.device)
                logits_loss = args.alpha * soft_loss + (1 - args.alpha) * hard_loss
                dist_loss = logits_loss + args.lambda_feature * feature_loss + args.lambda_contrast * contrastive_loss

            else:
                # ----- ours, normal batch -----
                dist_loss, hard_loss, soft_loss, feature_loss, contrastive_loss = multilayer_distillation_loss(
                    student_output, teacher_output, student_features, teacher_features,
                    target, args.temperature, args.alpha,
                    args.lambda_feature, args.lambda_contrast, criterion_hard, criterion_contrastive,
                    feature_projector=feature_projector)
                aux_loss = criterion_hard(student_aux, target)

                proto_loss = torch.zeros((), device=images.device)
                if teacher_prototypes is not None and args.lambda_proto > 0:
                    s_cls = student_features[:, 0, :]
                    t_cls = teacher_features[:, 0, :]
                    s_cls_proj_norm = F.normalize(feature_projector(s_cls), p=2, dim=-1)
                    for cls_id in range(7):
                        mask = (target == cls_id)
                        if mask.sum() > 0:
                            t_feat = t_cls[mask].mean(0).detach().float()
                            if cls_id not in teacher_prototypes:
                                teacher_prototypes[cls_id] = t_feat
                            else:
                                teacher_prototypes[cls_id] = (
                                    0.9 * teacher_prototypes[cls_id] + 0.1 * t_feat).detach()
                    proto_count = 0
                    for cls_id in range(7):
                        mask = (target == cls_id)
                        if mask.sum() > 0 and cls_id in teacher_prototypes:
                            proto_norm = F.normalize(
                                teacher_prototypes[cls_id].unsqueeze(0).to(s_cls_proj_norm.dtype), p=2, dim=-1)
                            s_mean = s_cls_proj_norm[mask].mean(0).unsqueeze(0)
                            proto_loss = proto_loss + (1 - F.cosine_similarity(s_mean, proto_norm)).squeeze()
                            proto_count += 1
                    if proto_count > 0:
                        proto_loss = proto_loss / proto_count

            # NA-MSAC (ours only)
            if na_msac_module is not None:
                features_list = [(x1_adapted, x1_adapted_flip), (x2_adapted, x2_adapted_flip),
                                 (x3_adapted, x3_adapted_flip)]
                spatial_attn_list = [
                    (attn_maps_orig[0], attn_maps_flip[0]),
                    (attn_maps_orig[1], attn_maps_flip[1]),
                    (attn_maps_orig[2], attn_maps_flip[2])
                ]
                na_msac_loss, na_msac_loss_dict = na_msac_module(features_list, spatial_attn_list)
            else:
                na_msac_loss = torch.zeros((), device=images.device)
                na_msac_loss_dict = {'na_msac_loss_total': 0.0}

            total_loss = (dist_loss + args.beta * aux_loss
                          + args.lambda_proto * proto_loss
                          + args.lambda_na_msac * na_msac_loss)
            total_loss = total_loss / args.accumulation_steps

        scaler.scale(total_loss).backward()
        if (i + 1) % args.accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if use_mixup_this_batch:
            _, predicted = student_output.max(1)
            acc1 = (lam * predicted.eq(target_a).sum().float()
                    + (1 - lam) * predicted.eq(target_b).sum().float()) / target.size(0) * 100.0
            acc1 = acc1.item()
        else:
            acc1 = accuracy(student_output, target, topk=(1,))[0].item()

        losses.update(total_loss.item() * args.accumulation_steps, images.size(0))
        hard_losses.update(hard_loss.item(), images.size(0))
        soft_losses.update(soft_loss.item(), images.size(0))
        feature_losses.update(feature_loss.item(), images.size(0))
        contrastive_losses.update(contrastive_loss.item(), images.size(0))
        proto_losses.update(proto_loss.item() if isinstance(proto_loss, torch.Tensor) else proto_loss,
                            images.size(0))
        na_msac_losses.update(na_msac_loss_dict['na_msac_loss_total'], images.size(0))
        top1.update(acc1, images.size(0))

        if i % 50 == 0:
            tag = args.kd_method.upper()
            if args.kd_method == 'kd':
                progress_msg = (f"  Epoch [{epoch}][{i}/{len(train_loader)}] [KD] "
                                f"Loss: {losses.avg:.4f} (Hard: {hard_losses.avg:.4f}, "
                                f"KL: {soft_losses.avg:.4f}) | Acc: {top1.avg:.2f}%")
            elif args.kd_method == 'fitnet':
                progress_msg = (f"  Epoch [{epoch}][{i}/{len(train_loader)}] [FitNet] "
                                f"Loss: {losses.avg:.4f} (Hard: {hard_losses.avg:.4f}, "
                                f"MSE: {feature_losses.avg:.5f}) | Acc: {top1.avg:.2f}%")
            elif args.kd_method == 'at':
                progress_msg = (f"  Epoch [{epoch}][{i}/{len(train_loader)}] [AT] "
                                f"Loss: {losses.avg:.4f} (Hard: {hard_losses.avg:.4f}, "
                                f"AT: {feature_losses.avg:.5f}) | Acc: {top1.avg:.2f}%")
            elif args.kd_method == 'simkd':
                progress_msg = (f"  Epoch [{epoch}][{i}/{len(train_loader)}] [SimKD] "
                                f"Loss: {losses.avg:.4f} (CE_via_T: {soft_losses.avg:.4f}, "
                                f"MSE: {feature_losses.avg:.5f}) | Acc: {top1.avg:.2f}%")
            else:
                progress_msg = (f"  Epoch [{epoch}][{i}/{len(train_loader)}] [OURS] "
                                f"Loss: {losses.avg:.4f} (Hard: {hard_losses.avg:.4f}, Soft: {soft_losses.avg:.4f}, "
                                f"Feat: {feature_losses.avg:.4f}, Contrast: {contrastive_losses.avg:.4f}, "
                                f"Proto: {proto_losses.avg:.4f}, NA-MSAC: {na_msac_losses.avg:.4f}) | "
                                f"Acc: {top1.avg:.2f}%")
            print(progress_msg)
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(progress_msg + '\n')

    return top1.avg, losses.avg


def validate(val_loader, model, criterion):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()
    with torch.no_grad():
        for images, target in val_loader:
            images, target = images.cuda(), target.cuda()
            output, _ = model(images)
            loss = criterion(output, target)
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
    return top1.avg, losses.avg


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

