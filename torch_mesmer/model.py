import torch
from torch import nn

from torch_mesmer.modules import SemanticHead, FeaturePyramidNetwork, Location2D, BackboneNetwork

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
            in_channels = 4
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
            backbone=self.backbone
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
