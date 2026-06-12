import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
from timm.models.layers import DropPath

try:
    from mobilefacenet import MobileFaceNet
    from vit_model import VisionTransformer, PatchEmbed
except ImportError:
    from .mobilefacenet import MobileFaceNet
    from .vit_model import VisionTransformer, PatchEmbed


def window_partition(x, window_size, h_w, w_w):
    B, H, W, C = x.shape
    x = x.view(B, h_w, window_size, w_w, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

class window(nn.Module):
    def __init__(self, window_size, dim):
        super(window, self).__init__()
        self.window_size = window_size
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        x = self.norm(x)
        shortcut = x
        h_w = int(torch.div(H, self.window_size).item())
        w_w = int(torch.div(W, self.window_size).item())
        x_windows = window_partition(x, self.window_size, h_w, w_w)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        return x_windows, shortcut

class WindowAttentionGlobal(nn.Module):
    def __init__(self, dim, num_heads, window_size, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        window_size = (window_size, window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = torch.div(dim, num_heads)
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, q_global, return_attention=False):
        B_, N, C = x.shape
        B = q_global.shape[0]
        head_dim = int(torch.div(C, self.num_heads).item())
        B_dim = int(torch.div(B_, B).item())
        kv = self.qkv(x).reshape(B_, N, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q_global = q_global.repeat(1, B_dim, 1, 1, 1)
        q = q_global.reshape(B_, self.num_heads, N, head_dim)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        if return_attention:
            attn_avg = attn.mean(dim=1)  # [B_, N, N]

            attn_map = attn_avg.sum(dim=1)  # [B_, N] - sum over each column

            H = W = self.window_size[0]
            attn_map = attn_map.view(B_, H, W)  # [B_, H, W]

            attn_map = attn_map.view(B, B_dim, H, W)  # [B, num_windows, H, W]
            attn_map = attn_map.mean(dim=1)  # [B, H, W]

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn_map
        return x

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

def _to_channel_last(x):
    return x.permute(0, 2, 3, 1)

def _to_query(x, N, num_heads, dim_head):
    B = x.shape[0]
    x = x.reshape(B, 1, N, num_heads, dim_head).permute(0, 1, 3, 2, 4)
    return x


class CrossScaleInteraction(nn.Module):
    def __init__(self, dim1=512, dim2=512, dim3=512, hidden_dim=64, num_heads=4):
        super().__init__()

        self.proj1 = nn.Linear(dim1, hidden_dim)
        self.proj2 = nn.Linear(dim2, hidden_dim)
        self.proj3 = nn.Linear(dim3, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=0.1
        )

        self.back_proj1 = nn.Linear(hidden_dim, dim1)
        self.back_proj2 = nn.Linear(hidden_dim, dim2)
        self.back_proj3 = nn.Linear(hidden_dim, dim3)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, t1, t2, t3):
        t1_proj = self.proj1(t1)  # [B, 196, 64]
        t2_proj = self.proj2(t2)  # [B, 196, 64]
        t3_proj = self.proj3(t3)  # [B, 49, 64]

        all_tokens = torch.cat([t1_proj, t2_proj, t3_proj], dim=1)  # [B, 441, 64]

        all_tokens_norm = self.norm(all_tokens)

        all_tokens_norm = all_tokens_norm.transpose(0, 1)  # [441, B, 64]
        enhanced, _ = self.cross_attn(
            all_tokens_norm,
            all_tokens_norm,
            all_tokens_norm
        )  # [441, B, 64]
        enhanced = enhanced.transpose(0, 1)  # [B, 441, 64]

        enhanced = enhanced + all_tokens

        N1, N2, N3 = t1.size(1), t2.size(1), t3.size(1)
        t1_enh, t2_enh, t3_enh = torch.split(enhanced, [N1, N2, N3], dim=1)

        t1_out = t1 + self.back_proj1(t1_enh)
        t2_out = t2 + self.back_proj2(t2_enh)
        t3_out = t3 + self.back_proj3(t3_enh)

        return t1_out, t2_out, t3_out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_reverse(windows, window_size, H, W, h_w, w_w):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, h_w, w_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class feedforward(nn.Module):
    def __init__(self, dim, window_size, mlp_ratio=4., act_layer=nn.GELU, drop=0., drop_path=0., layer_scale=None):
        super(feedforward, self).__init__()
        self.window_size = window_size
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm = nn.LayerNorm(dim)
    def forward(self, attn_windows, shortcut):
        B, H, W, C = shortcut.shape
        h_w = int(torch.div(H, self.window_size).item())
        w_w = int(torch.div(W, self.window_size).item())
        x = window_reverse(attn_windows, self.window_size, H, W, h_w, w_w)
        x = shortcut + self.mlp(self.norm(x))
        return x


class PosterV2_ResNet(nn.Module):
    def __init__(self, img_size=224, num_classes=9, dims=[64, 128, 256], window_size=[28, 14, 7], dropout=0.0, use_csi=True):
        super().__init__()
        self.use_csi = use_csi

        self.face_landback = MobileFaceNet([112, 112], 136)

        checkpoint_path = './models/pretrain/mobilefacenet_model_best.pth.tar'

        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            self.face_landback.load_state_dict(ckpt['state_dict'])
        except FileNotFoundError:
            print(f"[Warning] MobileFaceNet checkpoint not found at {checkpoint_path}. Using random init.")

        for p in self.face_landback.parameters():
            p.requires_grad = False
        self.last_face_conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        print(f"Loading ResNet18 Backbone with MS-Celeb pretrained weights...")
        resnet = models.resnet18(pretrained=False)

        pretrain_path = './models/pretrain/resnet18_msceleb.pth'
        try:
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            state_dict = checkpoint['state_dict']

            resnet_state_dict = resnet.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in resnet_state_dict and 'fc' not in k}
            resnet_state_dict.update(pretrained_dict)
            resnet.load_state_dict(resnet_state_dict)

            print(f"   Successfully loaded MS-Celeb pretrained weights from {pretrain_path}")
            print(f"   Loaded {len(pretrained_dict)} layers (excluding fc layer)")
        except FileNotFoundError:
            print(f"[Warning] MS-Celeb checkpoint not found at {pretrain_path}. Using random init.")

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1 # 64ch, 56x56
        self.layer2 = resnet.layer2 # 128ch, 28x28 -> Level 1
        self.layer3 = resnet.layer3 # 256ch, 14x14 -> Level 2
        self.layer4 = resnet.layer4 # 512ch, 7x7  -> Level 3

        self.adapt1 = nn.Sequential(nn.Conv2d(128, dims[0], 1, bias=False), nn.BatchNorm2d(dims[0]), nn.ReLU())
        self.adapt2 = nn.Sequential(nn.Conv2d(256, dims[1], 1, bias=False), nn.BatchNorm2d(dims[1]), nn.ReLU())
        self.adapt3 = nn.Sequential(nn.Conv2d(512, dims[2], 1, bias=False), nn.BatchNorm2d(dims[2]), nn.ReLU())

        self.window1 = window(window_size[0], dims[0])
        self.window2 = window(window_size[1], dims[1])
        self.window3 = window(window_size[2], dims[2])

        self.attn1 = WindowAttentionGlobal(dims[0], 2, window_size[0])
        self.attn2 = WindowAttentionGlobal(dims[1], 4, window_size[1])
        self.attn3 = WindowAttentionGlobal(dims[2], 8, window_size[2])

        self.ffn1 = feedforward(dims[0], window_size[0])
        self.ffn2 = feedforward(dims[1], window_size[1])
        self.ffn3 = feedforward(dims[2], window_size[2])

        self.embed_q = nn.Conv2d(dims[0], 512, 3, 2, 1) # 28 -> 14 [compress: 768->640->512]
        self.embed_k = nn.Conv2d(dims[1], 512, 3, 1, 1) # 14 -> 14 [compress: 768->640->512]
        self.embed_v = PatchEmbed(img_size=7, patch_size=1, in_c=dims[2], embed_dim=512)

        self.cross_scale_interaction = CrossScaleInteraction(
            dim1=512, dim2=512, dim3=512,
            hidden_dim=64, num_heads=4
        )

        self.VIT = VisionTransformer(embed_dim=512, depth=1, num_classes=num_classes, in_c=441, drop_ratio=dropout)

    def forward_backbone(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)
        return x1, x2, x3

    def forward_backbone_adapted(self, x):
        x1, x2, x3 = self.forward_backbone(x)
        x1_adapted = self.adapt1(x1)
        x2_adapted = self.adapt2(x2)
        x3_adapted = self.adapt3(x3)
        return x1_adapted, x2_adapted, x3_adapted

    def forward(self, x, return_features=False, return_all_layers=False, return_attention=False):
        # Face Branch
        x_face = F.interpolate(x, size=112)
        f1, f2, f3 = self.face_landback(x_face)
        f3 = self.last_face_conv(f3)

        f1, f2, f3 = _to_channel_last(f1), _to_channel_last(f2), _to_channel_last(f3)
        q1 = _to_query(f1, 28 * 28, 2, f1.shape[-1] // 2)
        q2 = _to_query(f2, 14 * 14, 4, f2.shape[-1] // 4)
        q3 = _to_query(f3, 7 * 7, 8, f3.shape[-1] // 8)

        # Image Branch (ResNet)
        x1, x2, x3 = self.forward_backbone(x)
        x1, x2, x3 = self.adapt1(x1), self.adapt2(x2), self.adapt3(x3)

        # Interaction
        w1, s1 = self.window1(x1)
        w2, s2 = self.window2(x2)
        w3, s3 = self.window3(x3)

        if return_attention:
            a1, attn_map1 = self.attn1(w1, q1, return_attention=True)
            a2, attn_map2 = self.attn2(w2, q2, return_attention=True)
            a3, attn_map3 = self.attn3(w3, q3, return_attention=True)

            o1 = self.ffn1(a1, s1)
            o2 = self.ffn2(a2, s2)
            o3 = self.ffn3(a3, s3)
        else:
            o1 = self.ffn1(self.attn1(w1, q1), s1)
            o2 = self.ffn2(self.attn2(w2, q2), s2)
            o3 = self.ffn3(self.attn3(w3, q3), s3)

        t1 = o1.view(x.shape[0], 28, 28, -1).permute(0, 3, 1, 2)
        t2 = o2.view(x.shape[0], 14, 14, -1).permute(0, 3, 1, 2)
        t3 = o3.view(x.shape[0], 7, 7, -1).permute(0, 3, 1, 2)

        t1 = self.embed_q(t1).flatten(2).transpose(1, 2)
        t2 = self.embed_k(t2).flatten(2).transpose(1, 2)
        t3 = self.embed_v(t3)  # PatchEmbed already does flatten and transpose internally

        if self.use_csi:
            t1, t2, t3 = self.cross_scale_interaction(t1, t2, t3)

        tokens = torch.cat([t1, t2, t3], dim=1)

        if return_attention:
            output, aux_output = self.VIT(tokens, return_features=False, return_all_layers=False)
            return output, aux_output, [attn_map1, attn_map2, attn_map3]
        elif return_features:
            if return_all_layers:
                output, aux_output, input_tokens, features, layer_features = self.VIT(tokens, return_features=True, return_all_layers=True)
                return output, aux_output, input_tokens, features, layer_features
            else:
                output, aux_output, input_tokens, features = self.VIT(tokens, return_features=True, return_all_layers=False)
                return output, aux_output, input_tokens, features
        else:
            output, aux_output = self.VIT(tokens, return_features=False, return_all_layers=False)
            return output, aux_output

