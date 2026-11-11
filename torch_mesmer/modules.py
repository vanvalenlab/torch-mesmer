import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_l, resnet50
from torchvision.models.efficientnet import EfficientNet_V2_L_Weights
from torchvision.models.resnet import ResNet50_Weights
import numpy as np

BACKBONE_CHANNELS = {
    'resnet50': {
        'C1': 64,
        'C2': 256,
        'C3': 512,
        'C4': 1024,
        'C5': 2048,
    },
    'efficientnetv2bl': {
        'C1': 32,
        'C2': 64,
        'C3': 96,
        'C4': 192,
        'C5': 512,
    },
}

class BackboneNetwork(nn.Module):

    '''
    A torchvision classification network, cut into the five stages C1..C5.

    ``self.backbone`` is a ``ModuleDict`` of the stages, run one after another;
    each stage's output is one level of backbone features.
    '''

    def __init__(self, backbone='efficientnetv2bl', use_imagenet=True):

        '''
            Args:
                backbone (str): one of ('efficientnetv2bl', 'resnet50'), case-insensitive.
                use_imagenet (bool): if True, start from torchvision's ImageNet weights
                    (downloaded on first use); otherwise from torchvision's random init.
        '''
        super().__init__()

        self.use_imagenet = use_imagenet
        self.backbone_name = backbone
        # The attribute is named ``backbone`` because the state_dict keys
        # (``bbnetwork.backbone.C1...``) are built from it.
        self.backbone = self._build_stages(backbone, use_imagenet)

    @staticmethod
    def _build_stages(name, use_imagenet):
        """Build the torchvision network ``name`` and return its stages as a ``ModuleDict``
        keyed ``'C1'``..``'C5'``, in the order they run."""

        _backbone = str(name).lower()

        if _backbone == 'efficientnetv2bl':
            if use_imagenet:
                model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
            else:
                model = efficientnet_v2_l()

            # EfficientNetV2-L features
            return nn.ModuleDict({
                'C1': model.features[0:1],      # Stage 0
                'C2': model.features[1:3],      # Stages 1-2
                'C3': model.features[3:4],      # Stage 3
                'C4': model.features[4:5],      # Stage 4
                'C5': model.features[5:7],      # Stages 5-6
            })

        elif _backbone == 'resnet50':
            if use_imagenet:
                model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                model = resnet50()

            return nn.ModuleDict({
                'C1': nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool),  # stride 4
                'C2': model.layer1,   # stride 4
                'C3': model.layer2,   # stride 8
                'C4': model.layer3,   # stride 16
                'C5': model.layer4,   # stride 32
            })

        else:
            raise ValueError("Unrecognized backbone.")

    def forward(self, x) -> dict:
        """Run ``x`` through the stages in order and keep every stage's output.

        Args:
            x (torch.Tensor): float image, channels first, shape ``(B, 3, H, W)``.

        Returns:
            dict[str, torch.Tensor]: ``{'C1': ..., 'C5': ...}``. Level ``Ck`` has shape
            ``(B, channels, H / 2**k, W / 2**k)``, except ResNet50's C1, which is
            already at ``H / 4`` because its stem ends in a max-pool.
        """
        backbone_features = {}

        current = x
        for level_name in self.backbone.keys():
            current = self.backbone[level_name](current)
            backbone_features[level_name] = current

        return backbone_features


class FeaturePyramidNetwork(nn.Module):
    """
    Build the feature pyramid network (FPN) using desired layers.
    """

    def __init__(
            self, 
            levels=['P3', 'P4', 'P5'], 
            backbone_levels = ['C3', 'C4', 'C5'],
            feature_size=256, 
            interpolation='bilinear',
            backbone='efficientnetv2bl'
        ):

        '''
        Args:
            levels (list[str]): list of strings describing the levels used to build the pyramid.
            backbone_levels (list[str]): list of strings describing the levels in the backbone network. Must have overlap with levels
            feature_size (int): used in the crown layers of the pyramid to generate any extra levels beyond those that match the backbone.
            interpolation (str): one of ('bilinear','nearest'), describing the interpolation mode. Usually 'bilinear'.
        '''

        super().__init__()
        
        # Validate that levels are in ascending order (P3, P4, P5)
        if levels:
            level_nums = [int(level[1:]) for level in levels]
            assert level_nums == sorted(level_nums), \
                f"Pyramid levels must be in ascending order (e.g., ['P3', 'P4', 'P5']), got {levels}"
        
        self.level_list = levels
        self.feature_size = feature_size
        self.backbone_levels = backbone_levels
        self.levels = nn.ModuleDict()
        self.crown = nn.ModuleDict()

        self.matched_levels = [level.replace('C','P') for level in self.backbone_levels]
        self.crown_levels = set(self.level_list) - set(self.matched_levels)
        self.crown_levels = sorted(list(self.crown_levels))

        channel_map =  BACKBONE_CHANNELS[backbone]
        
        # Build levels in reverse order (coarsest first)
        for i, matched_level in enumerate(reversed(self.matched_levels)):
            has_addition = (i > 0)  # All except coarsest receive top-down

            in_channels = channel_map[matched_level.replace('P','C')]
            
            self.levels[matched_level] = PyramidLevel(
                in_channels=in_channels,
                feature_size=feature_size, 
                has_addition=has_addition, 
                interpolation=interpolation
            )

        # Build crown in forward order (finest first)
        for crown_level in self.crown_levels:
            self.crown[crown_level] = nn.Conv2d(
                feature_size,
                feature_size,
                kernel_size=3,
                stride=2,
                padding=1
            )

    def forward(self, backbone_features) -> dict:
        '''
        Generate pyramid features from addition/upscale/conv with backbone features.

        Args:
            backbone_features (torch.Tensor): a dictionary containing the tensors from the backbone network with keys matching the backbone level they came from.

        '''
        pyramid_outputs = {}
        from_above = None

        
        # Build top-down: P5 → P4 → P3
        for pyr_level in reversed(self.matched_levels):
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

        for pyr_level in self.crown_levels:
            previous_level = f'P{int(pyr_level[1])-1}'
            output = self.crown[pyr_level](
                pyramid_outputs[previous_level]
            )
            pyramid_outputs[pyr_level] = output
        
        return pyramid_outputs

class PyramidLevel(nn.Module):

    '''
    Generates one level of the feature pyramid network that has a matched backbone level.
    '''

    def __init__(self, in_channels, feature_size=256, has_addition=False, interpolation='bilinear'):
        super().__init__()

        '''
        Args:
            feature_size: number of channels in the input layer
            has_addition (bool): whether or not there is an addition step from the pyramid level above. All but the coarsest get additions.
        '''
        
        self.lateral_conv = nn.Conv2d(in_channels, feature_size, kernel_size=1, stride=1)
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
    
    def __init__(self, 
                 n_classes=2,
                 feature_size=256,
                 pyramid_levels=['P3', 'P4', 'P5'],
                 crop_size=256,
                 n_dense=128,
                 interpolation='bilinear'):
        super().__init__()
        
        self.pyramid_levels = pyramid_levels
        self.n_levels = len(pyramid_levels)
        self.feature_size = feature_size
        self.crop_size = crop_size
        self.interpolation = interpolation
        
        # Build upsampling paths analytically — no forward pass needed
        self.level_processors = nn.ModuleDict()
        
        for level in pyramid_levels:
            level_num = int(level[1:])  # 'P3' -> 3
            feature_map_size = crop_size // (2 ** level_num)  # P3->32, P4->16, P5->8
            n_upsample = int(np.log2(crop_size / feature_map_size))  # how many 2x ups needed
            
            upsample_blocks = []
            for _ in range(n_upsample):
                upsample_blocks.extend([
                    nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode=interpolation)
                ])
            
            self.level_processors[level] = (
                nn.Sequential(*upsample_blocks) if upsample_blocks else nn.Identity()
            )
        
        # Fusion: concatenate all upsampled levels, reduce back to feature_size
        self.fusion_conv = nn.Conv2d(feature_size * self.n_levels, feature_size, kernel_size=1)
        self.fusion_bn = nn.BatchNorm2d(feature_size)
        self.fusion_relu = nn.ReLU(inplace=True)
        
        # Dense processing
        self.dense_conv = nn.Conv2d(feature_size, n_dense, kernel_size=1)
        self.bn = nn.BatchNorm2d(n_dense)
        self.relu = nn.ReLU(inplace=True)
        
        # Output
        self.output_conv = nn.Conv2d(n_dense, n_classes, kernel_size=1)
        
        if n_classes > 1:
            self.final_activation = nn.Softmax(dim=1)
        else:
            self.final_activation = nn.ReLU()

    def forward(self, pyramid_features):
        upsampled_features = []
        for level in self.pyramid_levels:
            upsampled_features.append(self.level_processors[level](pyramid_features[level]))
        
        x = torch.cat(upsampled_features, dim=1)
        x = self.fusion_relu(self.fusion_bn(self.fusion_conv(x)))
        x = self.relu(self.bn(self.dense_conv(x)))
        x = self.final_activation(self.output_conv(x))
        
        return x

class Location2D(torch.nn.Module):
    """Location Layer for 2D cartesian coordinate locations.

    Takes a channels-first ``(batch, channels, height, width)`` tensor and
    returns ``(batch, 2, height, width)``: the row coordinate then the column
    coordinate, each scaled to [0, 1].
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): ``(B, C, H, W)``; only its shape, dtype and device are used.

        Returns:
            torch.Tensor: ``(B, 2, H, W)``, with the dtype and device of ``inputs``. Channel 0 is
            the row index divided by ``H - 1``, channel 1 the column index divided by ``W - 1``.
            It is the same for every image in the batch, and carries no gradient.
        """
        batch, height, width = inputs.shape[0], inputs.shape[2], inputs.shape[3]

        rows = torch.arange(0, height, dtype=inputs.dtype, device=inputs.device)
        cols = torch.arange(0, width, dtype=inputs.dtype, device=inputs.device)
        rows = rows / torch.max(rows)
        cols = cols / torch.max(cols)

        row_map, col_map = torch.meshgrid(rows, cols, indexing='ij')  # each (H, W)
        location = torch.stack([row_map, col_map], dim=0)              # (2, H, W)
        return location.unsqueeze(0).repeat(batch, 1, 1, 1)
