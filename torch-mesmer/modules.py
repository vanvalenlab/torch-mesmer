import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_l
from torchvision.models.efficientnet import EfficientNet_V2_L_Weights
import numpy as np

class BackboneNetwork(nn.Module):

    def __init__(self, backbone='efficientnetv2bl', use_imagenet=True):
        super().__init__()
        
        self.use_imagenet = use_imagenet
        self.backbone=backbone
        self._extract_backbone()

    def _extract_backbone(self):

        _backbone = str(self.backbone).lower()
            
        if _backbone == 'efficientnetv2bl':
            if self.use_imagenet:
                model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
            else:
                model = efficientnet_v2_l()
            
            # EfficientNetV2-L features
            
            self.backbone = nn.ModuleDict({
                'C1': model.features[0:1],      # Stage 0
                'C2': model.features[1:3],      # Stages 1-2
                'C3': model.features[3:4],      # Stage 3
                'C4': model.features[4:5],      # Stage 4
                'C5': model.features[5:7],      # Stages 5-6
            })
        
        else:
            raise ValueError("Unrecognized backbone.")
        
        
    def forward(self, x) -> dict:
        """Extract backbone features sequentially through stages."""
        backbone_features = {}
        
        current = x
        for level_name in self.backbone.keys():
            current = self.backbone[level_name](current)
            backbone_features[level_name] = current
        
        return backbone_features

class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network following standard FPN convention."""

    def __init__(self, levels=['P3', 'P4', 'P5'], feature_size=256, interpolation='bilinear'):
        super().__init__()
        
        # Validate that levels are in ascending order (P3, P4, P5)
        if levels:
            level_nums = [int(level[1:]) for level in levels]
            assert level_nums == sorted(level_nums), \
                f"Pyramid levels must be in ascending order (e.g., ['P3', 'P4', 'P5']), got {levels}"
        
        self.level_list = levels
        self.feature_size = feature_size
        self.levels = nn.ModuleDict()
        
        # Build levels in reverse order (coarsest first)
        for i, curr_level in enumerate(reversed(levels)):
            has_addition = (i > 0)  # All except coarsest receive top-down
            
            self.levels[curr_level] = PyramidLevel(
                feature_size=feature_size, 
                has_addition=has_addition, 
                interpolation=interpolation
            )

    def forward(self, backbone_features) -> dict:
        pyramid_outputs = {}
        from_above = None
        
        # Build top-down: P5 → P4 → P3
        for pyr_level in reversed(self.level_list):
            backbone_level = pyr_level.replace('P', 'C')
            
            if backbone_level not in backbone_features:
                raise KeyError(
                    f"Backbone level {backbone_level} not found. "
                    f"Available: {list(backbone_features.keys())}"
                )
            
            output, upsampled = self.levels[pyr_level](
                backbone_features[backbone_level], 
                from_above
            )
            
            pyramid_outputs[pyr_level] = output
            from_above = upsampled
        
        return pyramid_outputs

class PyramidLevel(nn.Module):

    def __init__(self, feature_size=256, has_addition=False, interpolation='bilinear'):
        super().__init__()

        self.lateral_conv = nn.LazyConv2d(feature_size, kernel_size=1, stride=1)
        self.has_addition = has_addition
        self.upsample = nn.Upsample(scale_factor=2, mode=interpolation)
        self.smooth_conv = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

    def forward(self, x, from_above = None):

        """
        Args:
            x: backbone feature at this level
            from_above: optional feature from pyramid level above
        """

        # Lateral connection from backbone feature to pyramid
        lateral = self.lateral_conv(x)

        # Add if we have a top-down connection
        if self.has_addition and from_above is not None:
            lateral = lateral + from_above

        # Smooth with a 3x3 conv
        output = self.smooth_conv(lateral)

        # Upsample for next level down
        upsampled = self.upsample(output)

        return output, upsampled
    

# In modules.py - Replace SemanticHead class

class SemanticHead(nn.Module):
    """Semantic segmentation head that fuses multi-scale pyramid features.
    
    Automatically adapts to actual pyramid feature map sizes.
    """
    
    def __init__(self, 
                 n_classes=2,
                 feature_size=256,
                 pyramid_levels=['P3', 'P4', 'P5'],
                 crop_size=256,
                 n_dense=128,
                 interpolation='bilinear'):
        """
        Args:
            n_classes: Number of output classes
            feature_size: Number of channels in pyramid features
            pyramid_levels: List of pyramid level names (e.g., ['P3', 'P4', 'P5'])
            crop_size: Target output spatial size
            n_dense: Number of channels in dense layer
            interpolation: Upsampling interpolation mode
        
        Note: Spatial sizes are inferred dynamically during forward pass
        """
        super().__init__()
        
        self.pyramid_levels = pyramid_levels
        self.n_levels = len(pyramid_levels)
        self.feature_size = feature_size
        self.crop_size = crop_size
        self.interpolation = interpolation
        
        # We'll build the upsampling paths lazily on first forward pass
        self.level_processors = None
        self._initialized = False
        
        # Fusion layer: concatenate all levels then reduce to feature_size
        self.fusion_conv = nn.Conv2d(
            feature_size * self.n_levels,
            feature_size,
            kernel_size=1
        )
        self.fusion_bn = nn.BatchNorm2d(feature_size)
        self.fusion_relu = nn.ReLU(inplace=True)
        
        # Dense processing
        self.dense_conv = nn.Conv2d(feature_size, n_dense, kernel_size=1)
        self.bn = nn.BatchNorm2d(n_dense)
        self.relu = nn.ReLU(inplace=True)
        
        # Output head
        self.output_conv = nn.Conv2d(n_dense, n_classes, kernel_size=1)
        
        # Final activation
        if n_classes > 1:
            self.final_activation = nn.Softmax(dim=1)
        else:
            self.final_activation = nn.ReLU()
    
    def _build_upsampling_paths(self, pyramid_features):
        """Build upsampling paths based on actual pyramid feature sizes."""
        self.level_processors = nn.ModuleDict()
        
        for level in self.pyramid_levels:
            feat = pyramid_features[level]
            current_size = feat.shape[-1]  # Assume square feature maps
            
            # Calculate number of 2x upsamples needed
            n_upsample = int(np.log2(self.crop_size / current_size))
            
            if n_upsample < 0:
                raise ValueError(
                    f"Feature map for {level} is {current_size}×{current_size}, "
                    f"larger than crop_size {self.crop_size}. Cannot upsample."
                )
            
            # Build upsampling path
            upsample_blocks = []
            for _ in range(n_upsample):
                upsample_blocks.extend([
                    nn.Conv2d(self.feature_size, self.feature_size, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode=self.interpolation)
                ])
            
            if upsample_blocks:
                self.level_processors[level] = nn.Sequential(*upsample_blocks)
            else:
                self.level_processors[level] = nn.Identity()
        
        # Move to same device as the pyramid features
        device = next(iter(pyramid_features.values())).device
        self.level_processors = self.level_processors.to(device)
        
        self._initialized = True
        
        # Print info for debugging
        print(f"    SemanticHead initialized with pyramid feature sizes:")
        for level in self.pyramid_levels:
            size = pyramid_features[level].shape[-1]
            n_up = int(np.log2(self.crop_size / size))
            print(f"    {level}: {size}x{size} → {self.crop_size}x{self.crop_size} ({n_up} upsamples)")
        print()

    def forward(self, pyramid_features):
        """
        Args:
            pyramid_features: dict of pyramid features {'P3': tensor, 'P4': tensor, ...}
        
        Returns:
            Segmentation output at crop_size resolution
        """
        # Initialize upsampling paths on first forward pass
        if not self._initialized:
            self._build_upsampling_paths(pyramid_features)
        
        # Upsample each pyramid level to output resolution
        upsampled_features = []
        for level in self.pyramid_levels:
            feat = pyramid_features[level]
            upsampled = self.level_processors[level](feat)
            
            # Sanity check
            expected_size = self.crop_size
            actual_size = upsampled.shape[-1]
            if actual_size != expected_size:
                raise RuntimeError(
                    f"Upsampling failed for {level}: expected {expected_size}×{expected_size}, "
                    f"got {actual_size}×{actual_size}"
                )
            
            upsampled_features.append(upsampled)
        
        # Concatenate along channel dimension
        fused = torch.cat(upsampled_features, dim=1)
        
        # Reduce channels back to feature_size
        x = self.fusion_conv(fused)
        x = self.fusion_bn(x)
        x = self.fusion_relu(x)
        
        # Dense processing
        x = self.dense_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        # Output prediction
        x = self.output_conv(x)
        x = self.final_activation(x)
        
        return x

class Location2D(torch.nn.Module):
    """Location Layer for 2D cartesian coordinate locations.

    Args:
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        input_shape = inputs.shape
        input_device = inputs.device
        input_dtype = inputs.dtype
        
        # shapes of (, height) and (, width)
        x = torch.arange(0, input_shape[2], dtype=input_dtype, device=input_device)
        y = torch.arange(0, input_shape[3], dtype=input_dtype, device=input_device)

        # Detach after normalization to prevent gradient flow
        x = (x / torch.max(x)).detach()
        y = (y / torch.max(y)).detach()

        # makes the mesh
        loc_x, loc_y = torch.meshgrid(x, y, indexing='ij')

        # (2, H, W)
        loc = torch.stack([loc_x, loc_y], dim=0)

        # unsqueeze to add batch dimension and permute
        location = torch.unsqueeze(loc, dim=0)
        location = torch.permute(location, dims=[0, 2, 3, 1])

        # Detatch and tile to add back batch dimension (they are all the same)
        location = location.detach()
        location = torch.tile(location, [input_shape[0], 1, 1, 1])

        location = torch.permute(location, dims=[0, 3, 1, 2])

        return location