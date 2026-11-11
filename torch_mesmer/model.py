import torch
from torch import nn

from torch_mesmer.modules import SemanticHead, FeaturePyramidNetwork, Location2D, BackboneNetwork

# Architecture version, stored in every checkpoint as the ``arch_version`` buffer, so
# that weights are only ever loaded into the network they were trained with.
#   1: the network of the weights released on 2026-06-30 (torch-mesmer_2026-06-30.pth),
#      built with ``legacy=True``. Those files have no ``arch_version`` key. Its feature
#      pyramid passes the smoothed output down, builds P6 from P5 and has no ReLU before
#      P7.
#   2: the current network. The feature pyramid is wired as in deepcell: it passes the
#      merged lateral down, builds P6 from C5 and has a ReLU before P7.
ARCH_VERSION = 2
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
            in_channels = 4,
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

        ``legacy=True`` builds architecture version 1, the network of the weights
        released on 2026-06-30 (see ``ARCH_VERSION``), only to load those weights.
        Don't train with it.
        '''
        super().__init__()

        # Store configuration
        self.backbone = backbone
        self.use_imagenet = use_imagenet
        self.backbone_levels = backbone_levels
        self.pyramid_levels = pyramid_levels
        self.feature_size = feature_size
        self.interpolation = interpolation
        self.crop_size = crop_size
        self.n_semantic_classes = n_semantic_classes
        self.legacy = legacy

        # Build model
        self.location = Location2D()

        self.channel_conv = nn.Conv2d(
            in_channels=4,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        self.bbnetwork = BackboneNetwork(
            backbone=self.backbone, 
            use_imagenet=self.use_imagenet
        )

        self.fpn = FeaturePyramidNetwork(
            levels=self.pyramid_levels,
            feature_size=self.feature_size,
            interpolation=self.interpolation,
            backbone_levels=self.backbone_levels,
            backbone_channels=self.bbnetwork.out_channels,
            legacy=legacy,
        )
        
        # Create semantic heads
        semantic_heads = [
            SemanticHead(
                n_classes=i,
                feature_size=self.feature_size,
                pyramid_levels=self.pyramid_levels,
                crop_size=self.crop_size,
            )
            for i in self.n_semantic_classes
        ]

        self.semantic_heads = nn.ModuleList(semantic_heads)

        if not legacy:
            # Saved with the weights, so a checkpoint records which network it belongs to.
            self.register_buffer('arch_version', torch.tensor(ARCH_VERSION))

    @property
    def version(self):
        """Architecture version of this network: 1 with ``legacy=True``, otherwise ``ARCH_VERSION``."""
        return LEGACY_ARCH_VERSION if self.legacy else ARCH_VERSION

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
                hint = 'this code does not build that version'
            raise ValueError(
                f'Checkpoint is architecture version {found} but this PanopticNet is version '
                f'{self.version}; {hint}. Version 1 is the network of the weights released '
                f'on 2026-06-30.')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, x, format='channel_first'):
        
        if format == 'channels_last':
            x = x.permute(0, 3, 1, 2)
        
        location = self.location(x)
        x = torch.cat([x, location], dim=1)

        if x.shape[1] != 3:
            x = self.channel_conv(x)

        backbone_features = self.bbnetwork(x)
        pyramid_features = self.fpn(backbone_features)

        predictions = []
        for head in self.semantic_heads:
            predictions.append(head(pyramid_features))

        # Concatenate along the channel dimension; for Mesmer, (B, 8, H, W) in the
        # layout described in __init__ (nuclear channels 0-3, whole-cell 4-7).
        return torch.cat(predictions, dim=1)
