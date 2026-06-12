import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_flip_grid(w, h, device):
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float().to(device)
    grid = grid.unsqueeze(0).expand(1, -1, -1, -1)

    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1

    grid[:, 0, :, :] = -grid[:, 0, :, :]

    return grid


class NoiseAwareNativeAttention(nn.Module):

    def __init__(self,
                 num_classes=7,
                 feature_dims=[64, 128, 256],
                 feature_sizes=[28, 14, 7],
                 scale_weights=[0.2, 0.3, 0.5],
                 use_noise_aware=True,
                 noise_threshold=0.3,
                 use_class_aware=True,
                 single_scale_only=False):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dims = feature_dims
        self.feature_sizes = feature_sizes
        self.scale_weights = scale_weights
        self.use_noise_aware = use_noise_aware
        self.noise_threshold = noise_threshold
        self.use_class_aware = use_class_aware
        self.single_scale_only = single_scale_only

        self.fc1 = nn.Linear(feature_dims[0], num_classes)  # 64 -> 7
        self.fc2 = nn.Linear(feature_dims[1], num_classes)  # 128 -> 7
        self.fc3 = nn.Linear(feature_dims[2], num_classes)  # 256 -> 7

        self._init_weights()

        print(f"   [NA-MSAC] Initialized")
        print(f"   [NA-MSAC] Noise-aware: {use_noise_aware}")
        print(f"   [NA-MSAC] Single-scale-only: {single_scale_only}")
        print(f"   [NA-MSAC] Parameters: {self.count_parameters()} (FC layers)")
        print(f"   [NA-MSAC] Using real attention from WindowAttentionGlobal")
        print(f"   [NA-MSAC] Feature dims: {feature_dims}")
        print(f"   [NA-MSAC] Scale weights: {scale_weights}")

    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def generate_class_aware_attention(self, features, spatial_attn, fc_layer):
        B, C, H, W = features.shape

        fc_weights = fc_layer.weight  # [num_classes, C]

        fc_weights = fc_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        features_expanded = features.unsqueeze(1)  # [B, 1, C, H, W]

        class_response = (features_expanded * fc_weights).sum(dim=2)

        spatial_attn = spatial_attn.unsqueeze(1)

        class_attn = class_response * spatial_attn

        return class_attn

    def compute_noise_weight(self, attn_orig, attn_flip):
        diff = torch.abs(attn_orig - attn_flip).mean(dim=[1, 2, 3])  # [B]

        diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

        weights = torch.exp(-diff_norm / self.noise_threshold)

        return weights

    def ac_loss_single_scale(self, att_map1, att_map2, size):
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
        total_loss = 0.0
        loss_dict = {}

        fc_layers = [self.fc1, self.fc2, self.fc3]

        for i, ((x, x_flip), (attn, attn_flip), fc_layer, size, weight) in enumerate(
            zip(features_list, spatial_attn_list, fc_layers,
                self.feature_sizes, self.scale_weights)):

            single_scale_index = 0  # 0: 28x28, 1: 14x14, 2: 7x7

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
