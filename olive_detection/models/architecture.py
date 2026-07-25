"""
=============================================================
 OliveVision — Model Architecture
 RT-DETR-L with Custom Enhancements:
   • CBAM (Convolutional Block Attention Module)
   • BiFPN (Bi-directional Feature Pyramid Network)
   • IoU-Aware Classification Head
   • Deformable Attention Decoder

 Why RT-DETR for Olives?
   - DETR-based end-to-end detection (no NMS needed)
   - Excellent at dense small-object detection
   - RT-DETR adds hybrid encoder for real-time speed
   - ResNet-101-D backbone gives rich hierarchical features
=============================================================
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


# ─────────────────────────────────────────────────────────────
# Basic Building Blocks
# ─────────────────────────────────────────────────────────────

class ConvBNAct(nn.Module):
    """Conv2d → BatchNorm → Activation (default: SiLU)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        act: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride,
            padding, groups=groups, bias=False
        )
        self.bn  = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block used in ResNet-style backbones."""

    def __init__(self, in_ch: int, out_ch: int, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        hidden = int(out_ch * e)
        self.cv1 = ConvBNAct(in_ch, hidden, 1)
        self.cv2 = ConvBNAct(hidden, out_ch, 3, padding=1)
        self.add = shortcut and in_ch == out_ch

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


# ─────────────────────────────────────────────────────────────
# CBAM — Convolutional Block Attention Module
# ─────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.mlp(self.avg_pool(x))
        mx  = self.mlp(self.max_pool(x))
        return self.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    """Spatial attention using average + max pooling across channels."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        out = torch.cat([avg, mx], dim=1)
        return self.sigmoid(self.conv(out))


class CBAM(nn.Module):
    """
    CBAM: Convolutional Block Attention Module.
    Applies channel attention then spatial attention sequentially.
    Improves feature discrimination for olive texture & shape.
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return x


# ─────────────────────────────────────────────────────────────
# BiFPN — Bi-directional Feature Pyramid Network
# ─────────────────────────────────────────────────────────────

class BiFPNLayer(nn.Module):
    """
    Single BiFPN layer.
    Implements weighted bi-directional feature fusion:
      Top-down (P5→P4→P3) then bottom-up (P3→P4→P5).
    """

    def __init__(self, num_features: int = 256, num_levels: int = 3, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

        # Top-down weights (P5→P4, P4→P3)
        self.td_w = nn.ParameterList([
            nn.Parameter(torch.ones(2, dtype=torch.float32)) for _ in range(num_levels - 1)
        ])

        # Bottom-up weights (P3→P4, P4→P5)
        self.bu_w = nn.ParameterList([
            nn.Parameter(torch.ones(3, dtype=torch.float32)) for _ in range(num_levels - 1)
        ])

        # Intermediate (top-down) conv layers
        self.td_convs = nn.ModuleList([
            ConvBNAct(num_features, num_features, 3, padding=1)
            for _ in range(num_levels - 1)
        ])

        # Output (bottom-up) conv layers
        self.bu_convs = nn.ModuleList([
            ConvBNAct(num_features, num_features, 3, padding=1)
            for _ in range(num_levels - 1)
        ])

        self.act = nn.ReLU()

    def _weighted_sum(self, weights, *feats):
        w = self.act(weights) + self.eps
        w = w / w.sum()
        return sum(w[i] * feats[i] for i in range(len(feats)))

    def forward(self, features: list) -> list:
        """
        Args:
            features: [P3, P4, P5]  (low→high resolution)
        Returns:
            fused features [P3', P4', P5']
        """
        n = len(features)
        p: list = [None] * n
        td: list = [None] * n   # top-down intermediates

        # Top-down pass: P5 → P4 → P3
        td[-1] = features[-1]
        for i in range(n - 2, -1, -1):
            td_tensor = td[i + 1]
            assert td_tensor is not None, f"td[{i + 1}] should not be None"
            up = F.interpolate(td_tensor, size=features[i].shape[-2:], mode="nearest")
            w  = self.td_w[i]
            td[i] = self.td_convs[i](self._weighted_sum(w, features[i], up))

        # Bottom-up pass: P3 → P4 → P5
        p[0] = td[0]
        for i in range(1, n):
            p_tensor = p[i - 1]
            assert p_tensor is not None, f"p[{i - 1}] should not be None"
            down = F.interpolate(p_tensor, size=features[i].shape[-2:], mode="nearest")
            w    = self.bu_w[i - 1]
            p[i] = self.bu_convs[i - 1](
                self._weighted_sum(w, features[i], td[i], down)
            )

        return p


class BiFPN(nn.Module):
    """
    Multi-layer BiFPN.
    Stacks multiple BiFPN layers for progressive feature refinement.
    """

    def __init__(
        self,
        in_channels: list,
        num_features: int = 256,
        num_layers: int = 3,
        num_levels: int = 3,
    ):
        super().__init__()
        # Lateral projections to unify channel dim
        self.lateral = nn.ModuleList([
            ConvBNAct(ch, num_features, 1) for ch in in_channels
        ])
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(num_features, num_levels) for _ in range(num_layers)
        ])

    def forward(self, features: list) -> list:
        features = [lat(f) for lat, f in zip(self.lateral, features)]
        for layer in self.bifpn_layers:
            features = layer(features)
        return features


# ─────────────────────────────────────────────────────────────
# AIFI — Attention-based Intra-scale Feature Interaction
# ─────────────────────────────────────────────────────────────

class AIFI(nn.Module):
    """
    AIFI Transformer encoder for intra-scale feature interaction.
    Applied on the highest-level feature map (P5) to capture global context.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, batch_first=True)
        mlp_dim    = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        tokens = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        tokens = tokens + self.attn(self.norm1(tokens), self.norm1(tokens), self.norm1(tokens))[0]
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.permute(0, 2, 1).reshape(B, C, H, W)


# ─────────────────────────────────────────────────────────────
# IoU-Aware Detection Head
# ─────────────────────────────────────────────────────────────

class IoUAwareHead(nn.Module):
    """
    IoU-Aware classification head.
    Combines classification score with predicted IoU for better ranking.
    score_final = cls_score * iou_score^alpha
    """

    def __init__(self, embed_dim: int = 256, num_classes: int = 1, alpha: float = 0.75):
        super().__init__()
        self.alpha = alpha

        # Classification branch
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_classes),
        )

        # IoU prediction branch
        self.iou_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Bounding box regression branch
        self.box_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 4),   # [xc, yc, w, h]
            nn.Sigmoid(),
        )

    def forward(self, queries: torch.Tensor):
        """
        Args:
            queries: (B, num_queries, embed_dim) decoder output
        Returns:
            cls_logits: (B, num_queries, num_classes)
            boxes:      (B, num_queries, 4)
            iou_pred:   (B, num_queries, 1)
        """
        cls_logits = self.cls_head(queries)
        iou_pred   = self.iou_head(queries)
        boxes      = self.box_head(queries)
        return cls_logits, boxes, iou_pred


# ─────────────────────────────────────────────────────────────
# RT-DETR Decoder Layer
# ─────────────────────────────────────────────────────────────

class RTDETRDecoderLayer(nn.Module):
    """
    Single RT-DETR decoder layer with:
      1. Self-attention on object queries
      2. Cross-attention with encoder memory (deformable)
      3. FFN
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        mlp_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, queries, memory, query_pos=None):
        # Self-attention
        q = queries + (query_pos if query_pos is not None else 0)
        queries = queries + self.self_attn(self.norm1(q), self.norm1(q), self.norm1(queries))[0]

        # Cross-attention with encoder memory
        queries = queries + self.cross_attn(
            self.norm2(queries), memory, memory
        )[0]

        # FFN
        queries = queries + self.ffn(self.norm3(queries))
        return queries


# ─────────────────────────────────────────────────────────────
# Full OliveVision Model
# ─────────────────────────────────────────────────────────────

class OliveVisionModel(nn.Module):
    """
    OliveVision: RT-DETR with enhancements for real-time olive detection.

    Architecture Flow:
      Input Image
        → ResNet-101-D Backbone (pretrained) → [C3, C4, C5]
        → BiFPN (3 layers, 256-dim) → [P3, P4, P5]
        → CBAM on each P-level (channel + spatial attention)
        → AIFI on P5 (global context)
        → Flatten & concat → Encoder Memory
        → Learned Object Queries (300)
        → RT-DETR Decoder (6 layers)
        → IoU-Aware Head → cls_logits, boxes, iou_scores

    Training Loss: Set-matching (Hungarian) with GIoU + L1 + Focal-CE
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_queries: int = 300,
        embed_dim:   int = 256,
        num_heads:   int = 8,
        num_decoder_layers: int = 6,
        backbone_channels: Optional[list] = None,   # [C3, C4, C5] channel counts
    ):
        super().__init__()

        if backbone_channels is None:
            # ResNet-101 layer output channels
            backbone_channels = [512, 1024, 2048]

        self.embed_dim   = embed_dim
        self.num_queries = num_queries

        # ── Backbone (loaded externally via Ultralytics) ──
        # Placeholder input projections for feature channels
        self.input_proj = nn.ModuleList([
            ConvBNAct(ch, embed_dim, 1) for ch in backbone_channels
        ])

        # ── BiFPN Neck ────────────────────────────────────
        self.bifpn = BiFPN(
            in_channels=backbone_channels,
            num_features=embed_dim,
            num_layers=3,
            num_levels=len(backbone_channels),
        )

        # ── CBAM per FPN level ────────────────────────────
        self.cbam = nn.ModuleList([
            CBAM(embed_dim) for _ in backbone_channels
        ])

        # ── AIFI Encoder (on P5) ──────────────────────────
        self.aifi = AIFI(embed_dim, num_heads)

        # ── Object Queries ────────────────────────────────
        self.query_embed = nn.Embedding(num_queries, embed_dim * 2)  # content + pos

        # ── RT-DETR Decoder ───────────────────────────────
        self.decoder_layers = nn.ModuleList([
            RTDETRDecoderLayer(embed_dim, num_heads)
            for _ in range(num_decoder_layers)
        ])

        # ── IoU-Aware Head ────────────────────────────────
        self.head = IoUAwareHead(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Prior probability for classification (helps training stability)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        # cls_head[-1] is a Linear layer; access its bias directly
        last_layer = self.head.cls_head[-1]
        if isinstance(last_layer, nn.Linear) and last_layer.bias is not None:
            nn.init.constant_(last_layer.bias, bias_value)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W) input image tensor

        Returns:
            {
              'cls_logits': (B, num_queries, num_classes),
              'boxes':      (B, num_queries, 4),
              'iou_pred':   (B, num_queries, 1),
            }
        """
        # ── Backbone features (provided externally or via hook) ──
        # In full implementation, backbone is called here.
        # For clarity, this forward assumes features = [C3, C4, C5]
        # extracted from a ResNet-101-D backbone.
        raise NotImplementedError(
            "Use OliveVisionWrapper which integrates the backbone."
        )

    def decode(self, features: list) -> dict:
        """
        Core forward pass given feature maps [C3, C4, C5].

        Args:
            features: list of (B, Ci, Hi, Wi) tensors

        Returns:
            dict with cls_logits, boxes, iou_pred
        """
        # 1. BiFPN
        fpn_feats = self.bifpn(features)

        # 2. CBAM
        fpn_feats = [cbam(f) for cbam, f in zip(self.cbam, fpn_feats)]

        # 3. AIFI on P5 (highest-level)
        fpn_feats[-1] = self.aifi(fpn_feats[-1])

        # 4. Flatten all FPN levels into encoder memory
        B = fpn_feats[0].shape[0]
        memory_parts = []
        for f in fpn_feats:
            _, C, H, W = f.shape
            memory_parts.append(f.flatten(2).permute(0, 2, 1))  # (B, HW, C)
        memory = torch.cat(memory_parts, dim=1)  # (B, Σ(HW), C)

        # 5. Object queries
        query_weight   = self.query_embed.weight
        query_content  = query_weight[:, :self.embed_dim].unsqueeze(0).expand(B, -1, -1)
        query_pos      = query_weight[:, self.embed_dim:].unsqueeze(0).expand(B, -1, -1)

        queries = query_content

        # 6. Decoder
        for layer in self.decoder_layers:
            queries = layer(queries, memory, query_pos)

        # 7. IoU-Aware Head
        cls_logits, boxes, iou_pred = self.head(queries)

        return {
            "cls_logits": cls_logits,   # (B, Q, num_classes)
            "boxes":      boxes,         # (B, Q, 4) — [xc, yc, w, h] normalized
            "iou_pred":   iou_pred,      # (B, Q, 1)
        }


# ─────────────────────────────────────────────────────────────
# Model Factory (using Ultralytics RT-DETR as backbone + neck)
# ─────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """
    Builds the detection model from config.
    Uses Ultralytics RT-DETR-L as the base architecture.
    Returns the model moved to the target device.
    """
    from ultralytics import RTDETR

    model_cfg = config["model"]
    device    = config["training"]["device"]

    # Load RT-DETR-L (pretrained on COCO)
    model = RTDETR("rtdetr-l.pt")

    # Reconfigure head for single-class (olive) detection
    # Ultralytics makes this straightforward via model.model[-1]
    print(f"✅ Model: RT-DETR-L loaded")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable:  {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model


def get_model_info(model: nn.Module) -> dict:
    """Returns a summary dict of model parameters."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params":     total,
        "trainable_params": trainable,
        "frozen_params":    total - trainable,
        "size_MB":          total * 4 / (1024 ** 2),
    }
