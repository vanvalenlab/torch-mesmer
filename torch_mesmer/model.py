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
        For Mesmer, they have four semantic heads predicting different things:
            The first set is predicting nuclei:
                The first is predicting the inner distance transform of size 1 channels
                    (distance from center of an object to the nuclear boundary)
                The second semantic head is predicting the pixelwise transform of size 3 channels
                    (whether a pixel belongs to cell interior, boundary, or background)
            The second set is predicting cell boundaries:
                The first is predicting the inner distance transform of size 1 channels
                    (distance from center of an object to the cell boundary)
                The second semantic head is predicting the pixelwise transform of size 3 channels
                    (whether a pixel belongs to cell interior, boundary, or background)
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
            backbone_levels=self.backbone_levels
        )
        
        # Create semantic heads
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

        if x.shape[1] != 3:
            x = self.channel_conv(x)

        backbone_features = self.bbnetwork(x)
        pyramid_features = self.fpn(backbone_features)

        predictions = []
        for head in self.semantic_heads:
            predictions.append(head(pyramid_features))

        # Return concatenated along the first dimension. channels 0, 1, and 2 are discrete, channel 3 is continuous
        return torch.cat(predictions, dim=1)

if __name__ == '__main__':

    device = 'cuda:0'

    # Test with multi-scale heads
    model = PanopticNet(
        n_semantic_classes=[1,3,1,3],
        backbone='resnet50'
        ).to(device)

    test_tensor = torch.rand(8, 2, 256, 256).to(device)

    output = model(test_tensor, format='channels_first')

    print(f"Output shape: {output.shape}")  # Should be (8, 4, 256, 256)
