import warnings
warnings.filterwarnings("ignore")
import os
import sys
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models.PosterV2_9cls import PosterV2_ResNet
from models.PosterV2_Original_9 import pyramid_trans_expr2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

try:
    from noise_aware_native_attention import NoiseAwareNativeAttention
    NA_MSAC_AVAILABLE = True
except ImportError:
    print("Warning: NA-MSAC module not found. --lambda_na_msac will be disabled.")
    NA_MSAC_AVAILABLE = False
from torch.utils.data import DataLoader, Dataset
from thop import profile
from PIL import Image


class FocalLoss(nn.Module):
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
        else:
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
            similarity_loss = F.mse_loss(S_s, S_t)
            return similarity_loss
        else:
            similarity_matrix = torch.matmul(student_features, teacher_features.T) / self.temperature
            labels = torch.arange(batch_size).to(student_features.device)
            loss_s2t = F.cross_entropy(similarity_matrix, labels)
            loss_t2s = F.cross_entropy(similarity_matrix.T, labels)
            return (loss_s2t + loss_t2s) / 2.0


def mixup_data(x, y, alpha=0.2, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./data_preprocessing/FANE-divide-9folders/')
parser.add_argument('--teacher_path', type=str, default='./models/pretrain/fane-model_best.pth')
parser.add_argument('--checkpoint_FANE_path', type=str, default='./checkpoints/FANE_resnet_distill_model.pth')
parser.add_argument('--best_checkpoint_FANE_path', type=str, default='./checkpoints/resnet_distill_model_best.pth')
parser.add_argument('--resume', type=str, default='', help='Path to checkpoints to resume from')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--wd_backbone', default=1e-4, type=float,
                    help='Weight decay for ResNet backbone (small, to preserve transfer-learned features)')
parser.add_argument('--wd_head', default=1e-2, type=float,
                    help='Weight decay for ViT head + projector (large, to suppress head overfitting)')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='[legacy compat] uniform value used when differentiated WD is not applied')
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--use_mixup', default=True, type=bool)
parser.add_argument('--mixup_alpha', default=0.2, type=float,
                    help='Mixup alpha (lowered: 0.5->0.2, reduces double blurring with distillation soft labels)')
parser.add_argument('--mixup_prob', default=0.5, type=float,
                    help='Mixup probability (lowered: 0.8->0.5)')
parser.add_argument('--temperature', default=4.0, type=float,
                    help='Distillation Temperature (raised: 3.5->4.0, richer soft-label information)')
parser.add_argument('--alpha', default=0.6, type=float,
                    help='Distillation Loss Weight (raised: 0.5->0.6, relies more on teacher soft labels)')
parser.add_argument('--beta', default=0.5, type=float)
parser.add_argument('--label_smoothing', default=0.15, type=float)
parser.add_argument('--dropout', default=0.4, type=float)
parser.add_argument('--lambda_feature', default=0.7, type=float)
parser.add_argument('--lambda_contrast', default=0.0, type=float)
parser.add_argument('--lambda_proto', default=0.0, type=float,
                    help='Prototype distillation loss weight')
parser.add_argument('--lambda_proto_contrast', default=0.0, type=float,
                    help='Prototype inter-class InfoNCE loss weight')
parser.add_argument('--proto_contrast_temp', default=0.1, type=float,
                    help='Temperature for prototype InfoNCE loss')
parser.add_argument('--contrast_temperature', default=0.07, type=float)
parser.add_argument('--use_distance_contrast', default=True, type=bool)
parser.add_argument('--warmup_epochs', default=5, type=int)
parser.add_argument('--accumulation_steps', default=1, type=int)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--focal_gamma', default=2.5, type=float)
parser.add_argument('--lambda_na_msac', type=float, default=1.0,
                    help='NA-MSAC loss weight (recommended: 0.5-1.0, default: 1.0)')
parser.add_argument('--na_msac_noise_aware', action='store_true', default=True,
                    help='NA-MSAC: enable noise-aware mechanism (default: True)')
parser.add_argument('--na_msac_noise_threshold', type=float, default=0.3,
                    help='NA-MSAC: noise-aware threshold (default: 0.3)')
parser.add_argument('--na_msac_class_aware', action='store_true', default=False,
                    help='NA-MSAC: enable class-aware mode (default: False)')
parser.add_argument('--no_na_msac_class_aware', action='store_false', dest='na_msac_class_aware',
                    help='NA-MSAC: disable class-aware mode')
parser.add_argument('--use_csi', action='store_true', default=False,
                    help='Enable CSI (Cross-Scale Interaction)')
parser.add_argument('--no_csi', action='store_false', dest='use_csi',
                    help='Disable CSI')
parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name; self.fmt = fmt; self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class RecorderMeter(object):
    def __init__(self, total_epoch):
        self.total_epoch = total_epoch
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)
    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss; self.epoch_losses[idx, 1] = val_loss
        self.epoch_accuracy[idx, 0] = train_acc; self.epoch_accuracy[idx, 1] = val_acc
    def plot_curve(self, save_path):
        plt.figure(figsize=(12, 6))
        x = np.arange(self.total_epoch)
        plt.plot(x, self.epoch_accuracy[:, 0], label='Train Acc')
        plt.plot(x, self.epoch_accuracy[:, 1], label='Val Acc')
        plt.legend(); plt.grid(True)
        if save_path: plt.savefig(save_path)
        plt.close()


def multilayer_distillation_loss(student_logits, teacher_logits,
                                  student_features, teacher_features,
                                  labels, temperature, alpha,
                                  lambda_feature, lambda_contrast,
                                  criterion_hard, contrastive_criterion,
                                  feature_projector=None):
    # 1. Hard label loss
    hard_loss = criterion_hard(student_logits, labels)

    # 2. Soft label loss (KL divergence)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)

    # 3. Feature distillation
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
    print("=> Loading teacher model...")
    import sys
    original_recorder = sys.modules['__main__'].__dict__.get('RecorderMeter', None)
    original_recorder1 = sys.modules['__main__'].__dict__.get('RecorderMeter1', None)

    class TempRecorderMeter: pass
    class TempRecorderMeter1: pass

    sys.modules['__main__'].RecorderMeter = TempRecorderMeter
    sys.modules['__main__'].RecorderMeter1 = TempRecorderMeter1

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if original_recorder is not None:
        sys.modules['__main__'].RecorderMeter = original_recorder
    if original_recorder1 is not None:
        sys.modules['__main__'].RecorderMeter1 = original_recorder1

    teacher = pyramid_trans_expr2(img_size=224, num_classes=9)
    state_dict = checkpoint['state_dict']
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    teacher.load_state_dict(new_state_dict, strict=False)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    print(f"   Teacher model loaded! Accuracy: {checkpoint['best_acc']:.2f}%")
    return teacher


class RobustImageFolder(Dataset):
    def __init__(self, image_folder_dataset):
        self.dataset = image_folder_dataset
        self.valid_indices = []
        print(f"   Checking dataset integrity...")
        for idx in range(len(self.dataset)):
            try:
                img_path, _ = self.dataset.imgs[idx]
                img = Image.open(img_path); img.verify()
                self.valid_indices.append(idx)
            except Exception:
                print(f"   Skipping corrupted image: {self.dataset.imgs[idx][0]}")
        print(f"   Valid images: {len(self.valid_indices)}/{len(self.dataset)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        return self.dataset[self.valid_indices[idx]]


def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_distill(train_loader, student, teacher, criterion_hard, criterion_contrastive,
                  optimizer, epoch, scaler, args, log_file=None,
                  feature_projector=None, na_msac_module=None,
                  teacher_prototypes=None):
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

    for i, (images, target) in enumerate(train_loader):
        images, target = images.cuda(), target.cuda()

        if na_msac_module is not None:
            images_flip = torch.flip(images, dims=[3])

        use_mixup_this_batch = args.use_mixup and np.random.rand() < args.mixup_prob

        if use_mixup_this_batch:
            images, target_a, target_b, lam = mixup_data(
                images, target, alpha=args.mixup_alpha, device='cuda')
            if na_msac_module is not None:
                images_flip, _, _, _ = mixup_data(
                    images_flip, target, alpha=args.mixup_alpha, device='cuda')

        with torch.cuda.amp.autocast():
            if na_msac_module is not None:
                student_output, student_aux, _, student_features = student(images, return_features=True)
                x1_adapted, x2_adapted, x3_adapted = student.forward_backbone_adapted(images)
                x1_adapted_flip, x2_adapted_flip, x3_adapted_flip = student.forward_backbone_adapted(images_flip)
                with torch.no_grad():
                    _, _, attn_maps_orig = student(images, return_attention=True)
                    _, _, attn_maps_flip = student(images_flip, return_attention=True)
            else:
                student_output, student_aux, _, student_features = student(images, return_features=True)

            with torch.no_grad():
                teacher_output, _, _, teacher_features = teacher(images, return_features=True)

            if use_mixup_this_batch:
                hard_loss = mixup_criterion(criterion_hard, student_output, target_a, target_b, lam)

                soft_loss = F.kl_div(
                    F.log_softmax(student_output / args.temperature, dim=1),
                    F.softmax(teacher_output / args.temperature, dim=1),
                    reduction='batchmean'
                ) * (args.temperature ** 2)

                student_cls = student_features[:, 0, :]
                teacher_cls = teacher_features[:, 0, :]
                student_cls_projected = feature_projector(student_cls)
                student_cls_norm = F.normalize(student_cls_projected, p=2, dim=-1)
                teacher_cls_norm = F.normalize(teacher_cls, p=2, dim=-1)
                cls_loss = F.mse_loss(student_cls_norm, teacher_cls_norm)

                student_global = student_features[:, 1:, :].mean(dim=1)
                teacher_global = teacher_features[:, 1:, :].mean(dim=1)
                student_global_projected = feature_projector(student_global)
                student_global_norm = F.normalize(student_global_projected, p=2, dim=-1)
                teacher_global_norm = F.normalize(teacher_global, p=2, dim=-1)
                global_loss = F.mse_loss(student_global_norm, teacher_global_norm)
                feature_loss = 0.6 * cls_loss + 0.4 * global_loss

                contrastive_loss = criterion_contrastive(student_cls, teacher_cls)
                aux_loss = mixup_criterion(criterion_hard, student_aux, target_a, target_b, lam)
                proto_loss = torch.tensor(0.0).cuda()
                proto_contrast_loss = torch.tensor(0.0).cuda()

            else:
                dist_loss, hard_loss, soft_loss, feature_loss, contrastive_loss = multilayer_distillation_loss(
                    student_output, teacher_output,
                    student_features, teacher_features,
                    target, args.temperature, args.alpha,
                    args.lambda_feature, args.lambda_contrast,
                    criterion_hard, criterion_contrastive,
                    feature_projector=feature_projector
                )
                aux_loss = criterion_hard(student_aux, target)

                # Prototype distillation (positive alignment + inter-class InfoNCE)
                proto_loss = torch.tensor(0.0).cuda()
                proto_contrast_loss = torch.tensor(0.0).cuda()
                if teacher_prototypes is not None and (args.lambda_proto > 0 or args.lambda_proto_contrast > 0):
                    s_cls = student_features[:, 0, :]
                    t_cls = teacher_features[:, 0, :]
                    s_cls_proj = feature_projector(s_cls)
                    s_cls_proj_norm = F.normalize(s_cls_proj, p=2, dim=-1)
                    # Momentum update of prototypes
                    for cls_id in range(9):
                        mask = (target == cls_id)
                        if mask.sum() > 0:
                            t_feat = t_cls[mask].mean(0).detach().float()
                            if cls_id not in teacher_prototypes:
                                teacher_prototypes[cls_id] = t_feat
                            else:
                                teacher_prototypes[cls_id] = (
                                    0.9 * teacher_prototypes[cls_id] + 0.1 * t_feat
                                ).detach()
                    # Positive alignment: student feature -> correct-class prototype
                    if args.lambda_proto > 0:
                        proto_count = 0
                        for cls_id in range(9):
                            mask = (target == cls_id)
                            if mask.sum() > 0 and cls_id in teacher_prototypes:
                                proto_norm = F.normalize(
                                    teacher_prototypes[cls_id].unsqueeze(0).to(s_cls_proj_norm.dtype),
                                    p=2, dim=-1
                                )
                                s_mean = s_cls_proj_norm[mask].mean(0).unsqueeze(0)
                                proto_loss = proto_loss + (
                                    1 - F.cosine_similarity(s_mean, proto_norm)
                                ).squeeze()
                                proto_count += 1
                        if proto_count > 0:
                            proto_loss = proto_loss / proto_count
                    # Inter-class relation: InfoNCE
                    if args.lambda_proto_contrast > 0 and len(teacher_prototypes) >= 2:
                        proto_ids = sorted(teacher_prototypes.keys())
                        all_protos = torch.stack([
                            F.normalize(teacher_prototypes[pid].to(s_cls_proj_norm.dtype), p=2, dim=-1)
                            for pid in proto_ids
                        ])
                        contrast_count = 0
                        for cls_id in proto_ids:
                            mask = (target == cls_id)
                            if mask.sum() == 0:
                                continue
                            s_mean = s_cls_proj_norm[mask].mean(0)
                            sims = torch.mv(all_protos, s_mean) / args.proto_contrast_temp
                            pos_idx = proto_ids.index(cls_id)
                            label = torch.tensor([pos_idx], dtype=torch.long).cuda()
                            proto_contrast_loss = proto_contrast_loss + F.cross_entropy(sims.unsqueeze(0), label)
                            contrast_count += 1
                        if contrast_count > 0:
                            proto_contrast_loss = proto_contrast_loss / contrast_count

            if use_mixup_this_batch:
                logits_loss = args.alpha * soft_loss + (1 - args.alpha) * hard_loss
                dist_loss = logits_loss + args.lambda_feature * feature_loss + args.lambda_contrast * contrastive_loss

            if na_msac_module is not None:
                features_list = [(x1_adapted, x1_adapted_flip), (x2_adapted, x2_adapted_flip), (x3_adapted, x3_adapted_flip)]
                spatial_attn_list = [
                    (attn_maps_orig[0], attn_maps_flip[0]),  # scale 1: 28x28
                    (attn_maps_orig[1], attn_maps_flip[1]),  # scale 2: 14x14
                    (attn_maps_orig[2], attn_maps_flip[2])   # scale 3: 7x7
                ]
                na_msac_loss, na_msac_loss_dict = na_msac_module(features_list, spatial_attn_list)
            else:
                na_msac_loss = torch.tensor(0.0).cuda()
                na_msac_loss_dict = {'na_msac_loss_total': 0.0}

            total_loss = dist_loss + args.beta * aux_loss + args.lambda_na_msac * na_msac_loss + args.lambda_proto * proto_loss + args.lambda_proto_contrast * proto_contrast_loss
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
        proto_losses.update(proto_loss.item() if isinstance(proto_loss, torch.Tensor) else proto_loss, images.size(0))
        na_msac_losses.update(na_msac_loss_dict['na_msac_loss_total'], images.size(0))
        top1.update(acc1, images.size(0))

        if i % 50 == 0:
            progress_msg = (f"  Epoch [{epoch}][{i}/{len(train_loader)}] "
                            f"Loss: {losses.avg:.4f} (Hard: {hard_losses.avg:.4f}, "
                            f"Soft: {soft_losses.avg:.4f}, Feat: {feature_losses.avg:.4f}, "
                            f"Contrast: {contrastive_losses.avg:.4f}, Proto: {proto_losses.avg:.4f}"
                            + (f", NA-MSAC: {na_msac_losses.avg:.4f}" if na_msac_module is not None else "")
                            + f") | Acc: {top1.avg:.2f}%")
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


def main():
    if not os.path.exists('./checkpoints'): os.makedirs('./checkpoints')
    if not os.path.exists('./log_Fane'): os.makedirs('./log_Fane')

    args = parser.parse_args()

    import random, time
    if args.seed is not None:
        seed = args.seed
        seed_type = "manual"
    else:
        seed = int(time.time() * 1000) % (2 ** 32)
        seed_type = "auto"
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f'./log_Fane/train_v2_proto_seed{seed}_{timestamp}.log'
    curve_file = f'./log_Fane/train_v2_proto_seed{seed}_{timestamp}_curve.png'
    args.best_checkpoint_FANE_path = f'./checkpoints/resnet_distill_v2_proto_seed{seed}_best_{timestamp}.pth'

    def log_print(message):
        print(message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

    teacher = load_teacher_model(args.teacher_path)
    teacher = teacher.cuda()
    student = PosterV2_ResNet(img_size=224, num_classes=9, dropout=args.dropout, use_csi=args.use_csi)

    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    input_tensor = torch.randn(1, 3, 224, 224)
    flops, _ = profile(student, inputs=(input_tensor,), verbose=False)

    log_print(f"   Student Parameters: {trainable_params/1e6:.2f}M (trainable) + {frozen_params/1e6:.2f}M (frozen) = {total_params/1e6:.2f}M (total)")
    log_print(f"   Student FLOPs: {flops/1e9:.2f} G")
    log_print(f"   Log file: {log_file}")
    log_print(f"   Training started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"   - Random seed: {seed} ({seed_type})")
    log_print(f"   - Batch size: {args.batch_size}")
    log_print(f"   - Learning rate: {args.lr}")
    log_print(f"   - Weight decay backbone: {args.wd_backbone} [differentiated, small for backbone]")
    log_print(f"   - Weight decay head/proj: {args.wd_head} [differentiated, large for head]")
    log_print(f"   - Epochs: {args.epochs}")
    log_print(f"   - Warmup epochs: {args.warmup_epochs}")
    log_print(f"   - LR Scheduler: Warmup + Cosine Annealing")
    log_print(f"  -Distillation alpha: {args.alpha} ")
    log_print("-" * 80)

    student = student.cuda()

    # Feature projection layer
    student_embed_dim = 512
    teacher_embed_dim = 768
    feature_projector = nn.Linear(student_embed_dim, teacher_embed_dim).cuda()
    log_print(f"   - Feature projector: {student_embed_dim} → {teacher_embed_dim}")

    # Initialize NA-MSAC (noise-aware native attention)
    na_msac_module = None
    if args.lambda_na_msac > 0:
        if not NA_MSAC_AVAILABLE:
            log_print("   ERROR: NA-MSAC requested but module not available!")
            log_print("   Please check models/noise_aware_native_attention.py exists")
            exit(1)

        na_msac_module = NoiseAwareNativeAttention(
            num_classes=9,
            feature_dims=[64, 128, 256],  # dimensions of the adapted features
            feature_sizes=[28, 14, 7],
            scale_weights=[0.2, 0.3, 0.5],
            use_noise_aware=args.na_msac_noise_aware,
            noise_threshold=args.na_msac_noise_threshold,
            use_class_aware=args.na_msac_class_aware
        ).cuda()

        na_msac_params = sum(p.numel() for p in na_msac_module.parameters())
        log_print(f"   [NA-MSAC] Lambda: {args.lambda_na_msac}")
        log_print(f"   [NA-MSAC] Parameters: {na_msac_params}")
        log_print(f"   [NA-MSAC] Noise-aware:  {args.na_msac_noise_aware} (threshold={args.na_msac_noise_threshold})")
        log_print(f"   [NA-MSAC] Multi-scale:  True (28x28 + 14x14 + 7x7, weights={[0.2, 0.3, 0.5]})")
        log_print(f"   [NA-MSAC] Single-scale: False")

    # 3. Data preparation
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

    train_dataset = RobustImageFolder(datasets.ImageFolder(os.path.join(args.data, 'train'), transform=train_transform))
    val_dataset = RobustImageFolder(datasets.ImageFolder(os.path.join(args.data, 'valid'), transform=val_transform))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # 4. Optimizer -- differentiated weight_decay
    #
    #   Split parameters into three groups:
    #     Group A: ResNet backbone (stem + layer1~4 + adapt layers) -> wd = wd_backbone (small)
    #     Group B: ViT head + face branch + window attention etc.   -> wd = wd_head (large)
    #     Group C: feature_projector                                -> wd = wd_head (large)
    #
    backbone_param_names = set()
    backbone_modules = ['stem', 'layer1', 'layer2', 'layer3', 'layer4',
                        'adapt1', 'adapt2', 'adapt3']
    for module_name in backbone_modules:
        module = getattr(student, module_name, None)
        if module is not None:
            for name, _ in module.named_parameters():
                backbone_param_names.add(f"{module_name}.{name}")

    backbone_params = []
    head_params = []
    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = any(name.startswith(m + '.') for m in backbone_modules)
        if is_backbone:
            backbone_params.append(param)
        else:
            head_params.append(param)

    projector_params = list(feature_projector.parameters())

    param_groups = [
        {'params': backbone_params,  'weight_decay': args.wd_backbone, 'name': 'backbone'},
        {'params': head_params,       'weight_decay': args.wd_head,     'name': 'head'},
        {'params': projector_params,  'weight_decay': args.wd_head,     'name': 'projector'},
    ]

    if na_msac_module is not None:
        param_groups.append(
            {'params': list(na_msac_module.parameters()), 'weight_decay': args.wd_head, 'name': 'na_msac'}
        )

    log_print(f"   Param groups: backbone={len(backbone_params)} params (wd={args.wd_backbone}), "
              f"head={len(head_params)} params (wd={args.wd_head}), "
              f"projector={len(projector_params)} params (wd={args.wd_head})")

    optimizer = optim.AdamW(param_groups, lr=args.lr)
    criterion_focal = FocalLoss(alpha=1.0, gamma=args.focal_gamma).cuda()
    criterion_contrastive = ContrastiveLoss(
        temperature=args.contrast_temperature,
        use_distance=args.use_distance_contrast
    ).cuda()
    scaler = torch.cuda.amp.GradScaler()
    scheduler = get_lr_scheduler(optimizer, args.warmup_epochs, args.epochs)

    recorder = RecorderMeter(args.epochs)
    best_acc = 0.0
    start_epoch = args.start_epoch

    # Resume
    if args.resume:
        if os.path.isfile(args.resume):
            log_print(f"\n=> Loading checkpoints from '{args.resume}'")
            ckpt = torch.load(args.resume)
            if 'epoch' in ckpt and args.start_epoch == 0:
                start_epoch = ckpt['epoch'] + 1
                best_acc = ckpt['best_acc']
                student.load_state_dict(ckpt['state_dict'])
                # Note: v2 optimizer grouping differs from the original, so optimizer state cannot be loaded directly.
                # When resuming from an original-format checkpoints, skip loading the optimizer state.
                try:
                    optimizer.load_state_dict(ckpt['optimizer'])
                    log_print("   Loaded optimizer state.")
                except Exception as e:
                    log_print(f"   [Warning] Could not load optimizer state (param group mismatch): {e}")
                    log_print("   Optimizer will start fresh (LR schedule adjusts automatically).")
                if 'scheduler' in ckpt:
                    try:
                        scheduler.load_state_dict(ckpt['scheduler'])
                    except Exception:
                        pass
                if 'scaler' in ckpt:
                    scaler.load_state_dict(ckpt['scaler'])
                log_print(f"   Resuming from epoch {start_epoch}, best_acc={best_acc:.2f}%")
            else:
                best_acc = ckpt.get('best_acc', 0.0)
                student.load_state_dict(ckpt['state_dict'])
                log_print(f"   Loaded model weights, best_acc={best_acc:.2f}%")
        else:
            log_print(f"=> No checkpoints found at '{args.resume}'")

    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()
        log_print(f"   LR adjusted to: {optimizer.param_groups[0]['lr']:.6f} for epoch {start_epoch}")

    # 5. Training loop
    teacher_prototypes = {}  # class prototypes persisted across epochs {cls_id: tensor(768,)}
    for epoch in range(start_epoch, args.epochs):
        train_acc, train_loss = train_distill(
            train_loader, student, teacher, criterion_focal,
            criterion_contrastive, optimizer, epoch, scaler, args, log_file,
            feature_projector=feature_projector,
            na_msac_module=na_msac_module,
            teacher_prototypes=teacher_prototypes
        )
        val_acc, val_loss = validate(val_loader, student, criterion_focal)
        scheduler.step()
        recorder.update(epoch, train_loss, train_acc, val_loss, val_acc)
        recorder.plot_curve(curve_file)

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        current_lr = optimizer.param_groups[0]['lr']

        log_msg = (f"Epoch [{epoch}/{args.epochs}] "
                   f"LR: {current_lr:.6f} | "
                   f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                   f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                   f"Best Acc: {best_acc:.2f}%")
        log_print(log_msg)

        if is_best:
            ckpt_data = {
                'epoch': epoch,
                'state_dict': student.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'seed': seed,
                'seed_type': seed_type,
                'timestamp': timestamp,
                'config': vars(args),
                'recorder': {
                    'losses': recorder.epoch_losses,
                    'accuracy': recorder.epoch_accuracy
                }
            }
            torch.save(ckpt_data, args.best_checkpoint_FANE_path)
            log_print(f"  ✓ Best model saved to {args.best_checkpoint_FANE_path}")

        if (epoch + 1) % 10 == 0:
            ckpt_data = {
                'epoch': epoch,
                'state_dict': student.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'seed': seed,
                'seed_type': seed_type,
                'timestamp': timestamp,
                'config': vars(args),
                'recorder': {
                    'losses': recorder.epoch_losses,
                    'accuracy': recorder.epoch_accuracy
                }
            }
            torch.save(ckpt_data, args.checkpoint_FANE_path)
            log_print(f"  ✓ checkpoint_FANE saved to {args.checkpoint_FANE_path}")

    log_print("-" * 80)
    log_print(f"Training completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Best validation accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()



