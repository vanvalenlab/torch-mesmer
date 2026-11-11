import math

import torch
from torch import nn

from torch_mesmer.modules import (
    SemanticHead, FusedSemanticHead, FeaturePyramidNetwork, Location2D, BackboneNetwork,
    KERAS_BN_EPS, KERAS_BN_MOMENTUM, init_keras_style, level_number, set_batchnorm,
)

# Architecture version, stored in every checkpoint as the ``arch_version`` buffer, so
# that weights are only ever loaded into the network they were trained with.
#   1: the network of the weights released on 2026-06-30 (torch-mesmer_2026-06-30.pth),
#      built with ``legacy=True``. Those files have no ``arch_version`` key. Its feature
#      pyramid passes the smoothed output down, builds P6 from P5 and has no ReLU before
#      P7; it uses PyTorch's BatchNorm settings, and its semantic heads fuse every
#      pyramid level.
#   2: an intermediate network that was never released and is no longer built:
#      version 3's feature pyramid and BatchNorm settings with version 1's fused heads.
#   3: the current network. The feature pyramid is wired as in deepcell, BatchNorm
#      settings and initialisation follow Keras, and the semantic heads are Mesmer's
#      own, each reading P3 only. The P4/P5 3x3 convs and P6/P7, which nothing reads,
#      are not built.
ARCH_VERSION = 3
LEGACY_ARCH_VERSION = 1


def checkpoint_arch_version(state_dict):
    """Architecture version a state dict was saved from.

    Args:
        state_dict (dict): a ``PanopticNet`` state dict, e.g. from ``torch.load``.

    Returns:
        int: its ``arch_version`` entry, or ``LEGACY_ARCH_VERSION`` (1) if it has none,
        as for the 2026-06-30 weights.
    """
    version = state_dict.get('arch_version')
    return LEGACY_ARCH_VERSION if version is None else int(version)


class PanopticNet(nn.Module):
    def __init__(
            self,
            backbone='resnet50',
            feature_size=256,
            crop_size=256,
            use_imagenet=True,
            backbone_levels=['C3', 'C4', 'C5'],
            pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
            interpolation='bilinear',
            n_semantic_classes = [1, 3, 1, 3],
            in_channels = 2,
            legacy = False,
    ):
        '''
        For Mesmer (``n_semantic_classes=[1, 3, 1, 3]``) there are four semantic
        heads, **nuclear first** (the reverse of deepcell).
        Their outputs are concatenated into 8 channels:

            0     nuclear inner distance (ReLU): how far inside a nucleus a pixel is
            1-3   nuclear pixelwise (softmax): edge, interior, background
            4     whole-cell inner distance (ReLU)
            5-7   whole-cell pixelwise (softmax): edge, interior, background

        ``postprocess_utils.MESMER_HEAD_INDICES`` records which channels the
        watershed uses for each compartment.

        ``in_channels`` is the number of image channels (2 for Mesmer: nuclear,
        membrane). Two coordinate channels (``Location2D``) are added to the
        image, and a 1x1 conv maps the result to the 3 channels the backbone
        expects.

        ``crop_size``, and the side of every input image, must be a power of 2 and
        at least ``2 ** (coarsest pyramid level)`` (128 for Mesmer's P7), so that
        every pyramid level is at least 1 pixel.

        As in deepcell's PanopticNet, the heads read only the finest pyramid level
        (P3), so the coarser levels are not built; ``pyramid_levels`` still sets
        the smallest allowed input.

        ``legacy=True`` builds architecture version 1, the network of the weights
        released on 2026-06-30 (fused multi-level heads and the older pyramid
        wiring; see ``ARCH_VERSION``), only to load those weights. Don't train
        with it.
        '''
        super().__init__()

        if interpolation not in ('bilinear', 'nearest'):
            raise ValueError(f"interpolation must be 'bilinear' or 'nearest', got {interpolation!r}")

        # Store configuration
        self.backbone = backbone
        self.use_imagenet = use_imagenet
        self.backbone_levels = backbone_levels
        self.pyramid_levels = pyramid_levels
        self.feature_size = feature_size
        self.interpolation = interpolation
        self.crop_size = crop_size
        self.n_semantic_classes = n_semantic_classes
        self.in_channels = in_channels
        self.legacy = legacy
        self.min_size = 2 ** max(level_number(p) for p in pyramid_levels)
        self._check_size(crop_size, crop_size, what='crop_size')

        # Build model
        self.location = Location2D()

        self.channel_conv = nn.Conv2d(
            in_channels=in_channels + 2,  # image + 2 location channels
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        self.bbnetwork = BackboneNetwork(
            backbone=self.backbone, 
            use_imagenet=self.use_imagenet,
            keras_batchnorm=not legacy,
        )

        # deepcell: target_level = min(semantic_levels); the heads read that level only.
        head_level = min(self.pyramid_levels, key=level_number)

        self.fpn = FeaturePyramidNetwork(
            levels=self.pyramid_levels,
            feature_size=self.feature_size,
            interpolation=self.interpolation,
            backbone_levels=self.backbone_levels,
            backbone_channels=self.bbnetwork.out_channels,
            legacy=legacy,
            output_levels=None if legacy else [head_level],
        )

        self.semantic_heads = self._build_heads(head_level)

        if not legacy:
            self._init_keras_style()
            # Saved with the weights, so a checkpoint records which network it belongs to.
            self.register_buffer('arch_version', torch.tensor(ARCH_VERSION))

    def _build_heads(self, head_level):
        """One semantic head per entry of ``n_semantic_classes``, as an ``nn.ModuleList``.

        ``head_level`` is the pyramid level the heads read (the legacy fused heads read
        every level instead).
        """
        if self.legacy:
            semantic_heads = [
                FusedSemanticHead(
                    n_classes=i,
                    feature_size=self.feature_size,
                    pyramid_levels=self.pyramid_levels,
                    crop_size=self.crop_size,
                    interpolation=self.interpolation,
                )
                for i in self.n_semantic_classes
            ]
        else:
            semantic_heads = [
                SemanticHead(
                    n_classes=i,
                    level=head_level,
                    feature_size=self.feature_size,
                    n_upsample=level_number(head_level),
                    interpolation=self.interpolation,
                )
                for i in self.n_semantic_classes
            ]
        return nn.ModuleList(semantic_heads)

    def _init_keras_style(self):
        """Give every layer outside the backbone Keras's initial weights and BatchNorm
        settings, as in the released TF Mesmer. The backbone keeps its ImageNet weights,
        or torchvision's own init without them. Draws from the global random generator."""
        for part in (self.channel_conv, self.fpn, self.semantic_heads):
            init_keras_style(part)
        set_batchnorm(self.semantic_heads, eps=KERAS_BN_EPS, momentum=KERAS_BN_MOMENTUM)

    @property
    def version(self):
        """Architecture version of this network: 1 with ``legacy=True``, otherwise ``ARCH_VERSION``."""
        return LEGACY_ARCH_VERSION if self.legacy else ARCH_VERSION

    def _check_size(self, h, w, what='input'):
        # deepcell's PanopticNet requires square, power-of-2 inputs (panopticnet.py:170-182).
        # They must also be at least 2**(coarsest level), so every pyramid level is >= 1 pixel.
        if h != w or h < self.min_size or not math.log2(h).is_integer():
            raise ValueError(
                f'{what} must be square with a power-of-2 side of at least {self.min_size}, '
                f'got {h}x{w}')

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """``nn.Module.load_state_dict``, with a clear error when the checkpoint is for the
        other architecture version (old weights into the new network, or the reverse)."""
        found = checkpoint_arch_version(state_dict)
        if found != self.version:
            if found == LEGACY_ARCH_VERSION:
                hint = 'build the model with legacy=True to load it'
            elif found == ARCH_VERSION:
                hint = 'build the model with legacy=False to load it'
            else:
                hint = ('this code no longer builds that version (version 2 was an intermediate '
                        'network that was never released; see torch_mesmer.model.ARCH_VERSION)')
            raise ValueError(
                f'Checkpoint is architecture version {found} but this PanopticNet is version '
                f'{self.version}; {hint}. Version 1 is the network of the weights released '
                f'on 2026-06-30.')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, x, format='channels_first'):
        """Run the network on a batch of images.

        Args:
            x (torch.Tensor): float images, ``(B, in_channels, H, W)``, or
                ``(B, H, W, in_channels)`` with ``format='channels_last'``. They must be
                square, with a power-of-2 side of at least ``min_size``.
            format (str): the axis order of ``x``, ``'channels_first'`` or ``'channels_last'``.

        Returns:
            torch.Tensor: ``(B, sum(n_semantic_classes), H, W)``, always channels first:
            the heads' outputs, concatenated in order. For Mesmer that is the 8-channel
            layout described in ``__init__`` (nuclear channels 0-3, whole-cell 4-7).

        Raises:
            ValueError: for an unknown ``format``, or an input of the wrong shape or size.
        """
        if format == 'channels_last':
            x = x.permute(0, 3, 1, 2)
        elif format != 'channels_first':
            raise ValueError(f"format must be 'channels_first' or 'channels_last', got {format!r}")

        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f'expected input of shape (batch, {self.in_channels}, H, W) '
                             f'(channels first), got {tuple(x.shape)}')
        self._check_size(x.shape[2], x.shape[3])

        location = self.location(x)
        x = torch.cat([x, location], dim=1)
        x = self.channel_conv(x)

        backbone_features = self.bbnetwork(x)
        pyramid_features = self.fpn(backbone_features)

        predictions = []
        for head in self.semantic_heads:
            predictions.append(head(pyramid_features))

        return torch.cat(predictions, dim=1)
