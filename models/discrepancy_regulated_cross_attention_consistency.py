"""
Discrepancy-Regulated Cross-Attention Consistency (DR-CAC).

Enforces horizontal-flip consistency on multi-scale spatial attention maps.
The per-sample loss weight is regulated by the original-vs-flipped attention
discrepancy, so samples whose correspondence is unreliable contribute less.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_flip_grid(w, h, device):
    """
    Build a horizontal-flip sampling grid.

    Args:
        w: width
        h: height
        device: target device

    Returns:
        grid: [1, 2, H, W] flip grid
    """
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float().to(device)
    grid = grid.unsqueeze(0).expand(1, -1, -1, -1)

    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1

    grid[:, 0, :, :] = -grid[:, 0, :, :]

    return grid


class DiscrepancyRegulatedCrossAttentionConsistency(nn.Module):
    def __init__(self,
                 num_classes=7,
                 feature_dims=[64, 128, 256],
                 feature_sizes=[28, 14, 7],
                 scale_weights=[0.2, 0.3, 0.5],
                 use_noise_aware=True,
                 noise_threshold=0.3,
                 use_class_aware=True,
                 single_scale_only=False,
                 single_scale_index=0):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dims = feature_dims
        self.feature_sizes = feature_sizes
        self.scale_weights = scale_weights
        self.use_noise_aware = use_noise_aware
        self.noise_threshold = noise_threshold
        self.use_class_aware = use_class_aware
        self.single_scale_only = single_scale_only
        self.single_scale_index = single_scale_index

        self.fc1 = nn.Linear(feature_dims[0], num_classes)  # 64 -> 7
        self.fc2 = nn.Linear(feature_dims[1], num_classes)  # 128 -> 7
        self.fc3 = nn.Linear(feature_dims[2], num_classes)  # 256 -> 7

        self._init_weights()

        print(f"   [NA-MSAC] Initialized")
        print(f"   [NA-MSAC] Noise-aware: {use_noise_aware}")
        print(f"   [NA-MSAC] Single-scale-only: {single_scale_only}")
        if single_scale_only:
            print(f"   [NA-MSAC] Single-scale index: {single_scale_index} "
                  f"({feature_sizes[single_scale_index]}x{feature_sizes[single_scale_index]})")
        print(f"   [NA-MSAC] Parameters: {self.count_parameters()} (FC layers)")
        print(f"   [NA-MSAC] Using real attention from WindowAttentionGlobal")
        print(f"   [NA-MSAC] Feature dims: {feature_dims}")
        print(f"   [NA-MSAC] Scale weights: {scale_weights}")

    def _init_weights(self):
        """Xavier initialization."""
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def count_parameters(self):
        """Count the number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def generate_class_aware_attention(self, features, spatial_attn, fc_layer):
        """
        Build class-aware attention maps.

        Args:
            features: [B, C, H, W] - feature maps
            spatial_attn: [B, H, W] - spatial attention (from WindowAttentionGlobal)
            fc_layer: FC layer used to produce per-class responses

        Returns:
            class_attn: [B, num_classes, H, W] - class-specific attention maps
        """
        B, C, H, W = features.shape

        fc_weights = fc_layer.weight  # [num_classes, C]

        # fc_weights: [num_classes, C] -> [1, num_classes, C, 1, 1]
        fc_weights = fc_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        features_expanded = features.unsqueeze(1)  # [B, 1, C, H, W]

        class_response = (features_expanded * fc_weights).sum(dim=2)

        # spatial_attn: [B, H, W] -> [B, 1, H, W]
        spatial_attn = spatial_attn.unsqueeze(1)

        class_attn = class_response * spatial_attn

        return class_attn

    def compute_noise_weight(self, attn_orig, attn_flip):
        """
        Compute noise-aware sample weights.

        Args:
            attn_orig: attention of the original image [B, num_classes, H, W]
            attn_flip: attention of the flipped image [B, num_classes, H, W]

        Returns:
            weights: per-sample weights [B]
        """
        diff = torch.abs(attn_orig - attn_flip).mean(dim=[1, 2, 3])  # [B]

        diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

        weights = torch.exp(-diff_norm / self.noise_threshold)

        return weights

    def ac_loss_single_scale(self, att_map1, att_map2, size):
        """
        Attention-consistency loss for a single scale.

        Args:
            att_map1: attention map of the original image [B, num_classes, H, W]
            att_map2: attention map of the flipped image [B, num_classes, H, W]
            size: spatial size of the feature map

        Returns:
            loss: AC loss
            sample_weights: per-sample weights (when noise-aware is enabled)
        """
        B = att_map1.size(0)

        grid = generate_flip_grid(size, size, att_map1.device)
        flip_grid = grid.expand(B, -1, -1, -1)
        flip_grid = flip_grid.permute(0, 2, 3, 1)

        att_map2_flip = F.grid_sample(att_map2, flip_grid,
                                       mode='bilinear',
                                       padding_mode='border',
                                       align_corners=True)

        if self.use_noise_aware:
            sample_weights = self.compute_noise_weight(att_map1, att_map2_flip)
        else:
            sample_weights = torch.ones(B, device=att_map1.device)

        mse_per_sample = F.mse_loss(att_map1, att_map2_flip, reduction='none')
        mse_per_sample = mse_per_sample.mean(dim=[1, 2, 3])  # [B]

        weighted_loss = (mse_per_sample * sample_weights).mean()

        return weighted_loss, sample_weights

    def forward(self, features_list, spatial_attn_list):
        """
        Compute the multi-scale class-aware attention-consistency loss.

        Args:
            features_list: [(x1, x1_flip), (x2, x2_flip), (x3, x3_flip)]
                          each element is (original features, flipped features)
            spatial_attn_list: [(attn1, attn1_flip), (attn2, attn2_flip), (attn3, attn3_flip)]
                              each element is (original spatial attention,
                              flipped spatial attention)

        Returns:
            total_loss: weighted total AC loss
            loss_dict: per-scale losses (for logging)
        """
        total_loss = 0.0
        loss_dict = {}

        fc_layers = [self.fc1, self.fc2, self.fc3]

        for i, ((x, x_flip), (attn, attn_flip), fc_layer, size, weight) in enumerate(
            zip(features_list, spatial_attn_list, fc_layers,
                self.feature_sizes, self.scale_weights)):

            single_scale_index = self.single_scale_index

            if self.single_scale_only and i != single_scale_index:
                loss_dict[f'na_msac_loss_scale{i + 1}'] = 0.0
                loss_dict[f'na_msac_weight_scale{i + 1}'] = 1.0
                continue

            if self.use_class_aware:
                class_attn = self.generate_class_aware_attention(x, attn, fc_layer)
                class_attn_flip = self.generate_class_aware_attention(x_flip, attn_flip, fc_layer)
            else:
                class_attn = attn.unsqueeze(1)
                class_attn_flip = attn_flip.unsqueeze(1)

            loss, sample_weights = self.ac_loss_single_scale(class_attn, class_attn_flip, size)

            effective_weight = 1.0 if self.single_scale_only else weight
            total_loss += effective_weight * loss
            loss_dict[f'na_msac_loss_scale{i+1}'] = loss.item()
            loss_dict[f'na_msac_weight_scale{i+1}'] = sample_weights.mean().item()

        loss_dict['na_msac_loss_total'] = total_loss.item()

        return total_loss, loss_dict
