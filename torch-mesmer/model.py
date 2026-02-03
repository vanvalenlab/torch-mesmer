import torch
from torch import nn
from math import log2

from modules import SemanticHead, FeaturePyramidNetwork, Location2D, BackboneNetwork

# In model.py - Update PanopticNet class

class PanopticNet(nn.Module):
    def __init__(
            self,
            backbone='efficientnetv2bl',
            feature_size=256,
            crop_size=256,
            use_imagenet=True,
            backbone_levels=['C1','C2','C3', 'C4', 'C5'],
            pyramid_levels = ['P3', 'P4', 'P5'],
            interpolation='bilinear',
            n_semantic_classes = [1,1,2]
    ):
        
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

        self.bbnetwork = BackboneNetwork(
            backbone=self.backbone, 
            use_imagenet=self.use_imagenet
        )

        self.fpn = FeaturePyramidNetwork(
            levels=self.pyramid_levels,
            feature_size=self.feature_size,
            interpolation=self.interpolation
        )
        
        # Create semantic heads - pyramid shapes inferred automatically
        semantic_heads = [
            SemanticHead(
                n_classes=n_classes,
                feature_size=self.feature_size,
                pyramid_levels=self.pyramid_levels,
                crop_size=self.crop_size,
                interpolation=self.interpolation
            ) 
            for n_classes in self.n_semantic_classes
        ]

        self.semantic_heads = nn.ModuleList(semantic_heads)

    def forward(self, x, format='channel_first'):
        if format == 'channels_last':
            x = x.permute(0, 3, 1, 2)
        
        location = self.location(x)
        x = torch.cat([x, location], dim=1)

        backbone_features = self.bbnetwork(x)
        pyramid_features = self.fpn(backbone_features)

        predictions = []
        for head in self.semantic_heads:
            predictions.append(head(pyramid_features))

        return torch.cat(predictions, dim=1)

if __name__ == '__main__':

    device = 'cuda:0'

    # Test with multi-scale heads
    model = PanopticNet(
        n_semantic_classes=[1,1,2],
        use_multiscale_heads=True
    ).to(device)

    test_tensor = torch.rand(8, 256, 256, 1).to(device)

    output = model(test_tensor, format='channels_last')

    print(f"Output shape: {output.shape}")  # Should be (8, 4, 256, 256)