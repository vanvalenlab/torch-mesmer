import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_l, resnet50
from torchvision.models.efficientnet import EfficientNet_V2_L_Weights
from torchvision.models.resnet import ResNet50_Weights
import numpy as np


def level_number(name):
    """'P10' -> 10, 'C3' -> 3."""
    return int(name[1:])


class _SplitMBConv(nn.Module):
    """The second half of a torchvision MBConv whose first half (the expand conv) was tapped.

    Only valid for blocks without a residual connection (stride 2), where
    ``MBConv.forward(x) == block(x)``.
    """

    def __init__(self, mbconv):
        super().__init__()
        assert not mbconv.use_res_connect, 'can only split an MBConv without a residual connection'
        self.rest = mbconv.block[1:]

    def forward(self, x):
        return self.rest(x)


class BackboneNetwork(nn.Module):

    '''
    A torchvision classification network, cut into the five stages C1..C5.

    ``self.backbone`` is a ``ModuleDict`` of the stages, run one after another;
    each stage's output is one level of backbone features.

    ``out_channels`` maps each level (C1..C5) to its channel count. It is
    measured with a dummy forward pass rather than hard-coded, so it is right
    for either backbone.
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

        self.out_channels = self._measure_channels()

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

            # Tap points match deepcell's Keras EfficientNetV2 taps
            # (deepcell-tf/deepcell/utils/backbone_utils.py:418-432):
            #   C1 block1b_add, C2 block2c_add, C3 block4a_expand_activation,
            #   C4 block6a_expand_activation, C5 top_activation.
            # torchvision's features[i] is Keras stage i (features[0] = stem, features[8] = top conv),
            # with the same blocks per stage. The two "expand_activation" taps sit inside the first
            # block of a stage, after its 1x1 expand conv; those blocks have stride 2 and no residual,
            # so they can be cut there.
            f = model.features
            return nn.ModuleDict({
                'C1': nn.Sequential(f[0], f[1][0:2]),                        # stride 2
                'C2': nn.Sequential(f[1][2:], f[2][0:3]),                    # stride 4
                'C3': nn.Sequential(f[2][3:], f[3], f[4][0].block[0]),       # stride 8
                'C4': nn.Sequential(_SplitMBConv(f[4][0]), f[4][1:], f[5],
                                    f[6][0].block[0]),                       # stride 16
                'C5': nn.Sequential(_SplitMBConv(f[6][0]), f[6][1:], f[7], f[8]),  # stride 32
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

    @torch.no_grad()
    def _measure_channels(self):
        """Channel count of each level, from a forward pass in eval mode (running stats untouched)."""
        was_training = self.training
        self.eval()
        try:
            feats = self(torch.zeros(1, 3, 64, 64))
        finally:
            self.train(was_training)
        return {name: t.shape[1] for name, t in feats.items()}

    def forward(self, x) -> dict:
        """Run ``x`` through the stages in order and keep every stage's output.

        Args:
            x (torch.Tensor): float image, channels first, shape ``(B, 3, H, W)``.

        Returns:
            dict[str, torch.Tensor]: ``{'C1': ..., 'C5': ...}``. Level ``Ck`` has shape
            ``(B, out_channels['Ck'], H / 2**k, W / 2**k)``, except ResNet50's C1, which is
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

    Follows deepcell-tf ``fpn.py`` (``__create_pyramid_level`` and
    ``__create_pyramid_features``):

    - Matched levels (one per backbone level), coarsest first:
      ``merged = lateral_1x1(C) + upsample(merged from the level above)``, and
      ``P = smooth_3x3(merged)``. The *merged* tensor goes down the pyramid, not
      the smoothed one.
    - Crown levels above the backbone (P6, P7, ...): the first is a 3x3 stride-2
      conv on the coarsest backbone feature (C5); each further one applies
      ReLU and then a 3x3 stride-2 conv to the level below it.

    ``legacy=True`` rebuilds the different wiring of the weights released on
    2026-06-30 (torch-mesmer_2026-06-30.pth), which were trained before the
    wiring above was matched to deepcell: the smoothed output goes down the
    pyramid, P6 comes from P5, and there is no ReLU before P7. Use it only to
    load those weights.

    ``levels`` holds the matched ``PyramidLevel`` modules and ``crown`` the
    crown convs, each keyed by level name (both names are part of the
    state_dict keys). ``level_names`` is the list of requested level names.
    """

    def __init__(
            self,
            levels=['P3', 'P4', 'P5'],
            backbone_levels = ['C3', 'C4', 'C5'],
            feature_size=256,
            interpolation='bilinear',
            backbone_channels=None,
            legacy=False,
        ):

        '''
        Args:
            levels (list[str]): names of the pyramid levels, finest first, e.g. ``['P3', ..., 'P7']``.
                Those above the coarsest backbone level (the crown) must follow it without gaps.
            backbone_levels (list[str]): the backbone levels to build matched pyramid levels from,
                e.g. ``['C3', 'C4', 'C5']``.
            feature_size (int): number of channels in every pyramid level.
            interpolation (str): one of ('bilinear','nearest'), describing the interpolation mode. Usually 'bilinear'.
            backbone_channels (dict): channel count of each backbone level (``BackboneNetwork.out_channels``).
            legacy (bool): reproduce the wiring of the 2026-06-30 weights (see the class docstring).
        '''

        super().__init__()

        if backbone_channels is None:
            raise ValueError('backbone_channels is required (use BackboneNetwork.out_channels)')

        self.level_names = levels
        self.feature_size = feature_size
        self.backbone_levels = sorted(backbone_levels, key=level_number)
        self.legacy = legacy
        self.matched_levels, self.crown_levels = self._plan_levels(levels, self.backbone_levels)

        # Built in this order (coarsest matched level first, then the crown) so that
        # seeded runs draw the same initial weights for each layer.
        self.levels = self._build_matched_levels(backbone_channels, interpolation)
        self.crown = self._build_crown(backbone_channels)

    @staticmethod
    def _plan_levels(levels, backbone_levels):
        """Check the requested levels and work out which ones to build.

        Args:
            levels (list[str]): requested pyramid level names (see ``__init__``).
            backbone_levels (list[str]): backbone level names, finest first.

        Returns:
            (list[str], list[str]): finest first, the matched levels (one per backbone level)
            and the crown levels.
        """
        level_nums = [level_number(level) for level in levels]
        assert level_nums == sorted(level_nums), \
            f"Pyramid levels must be in ascending order (e.g., ['P3', 'P4', 'P5']), got {levels}"

        matched_levels = [level.replace('C', 'P') for level in backbone_levels]
        # Sorted by number, not as strings, so that 'P10' comes after 'P9'.
        crown_levels = sorted(set(levels) - set(matched_levels), key=level_number)

        # Crown levels must continue the pyramid without gaps: P6, P7, ...
        top = level_number(matched_levels[-1])
        expected = [f'P{top + 1 + i}' for i in range(len(crown_levels))]
        if crown_levels != expected:
            raise ValueError(f'Pyramid levels above {matched_levels[-1]} must be contiguous; '
                             f'expected {expected}, got {crown_levels}')

        return matched_levels, crown_levels

    def _build_matched_levels(self, backbone_channels, interpolation):
        """One ``PyramidLevel`` per backbone level, as a ``ModuleDict`` built coarsest first."""
        matched = nn.ModuleDict()
        for i, matched_level in enumerate(reversed(self.matched_levels)):
            matched[matched_level] = PyramidLevel(
                in_channels=backbone_channels[matched_level.replace('P', 'C')],
                feature_size=self.feature_size,
                has_addition=i > 0,  # every level but the coarsest adds in the one above
                interpolation=interpolation,
                legacy=self.legacy,
            )
        return matched

    def _build_crown(self, backbone_channels):
        """The 3x3 stride-2 conv of each crown level, as a ``ModuleDict`` built finest first."""
        crown = nn.ModuleDict()
        for i, crown_level in enumerate(self.crown_levels):
            if i == 0 and not self.legacy:
                # The first crown level reads the coarsest backbone feature (P6 from C5).
                in_channels = backbone_channels[self.backbone_levels[-1]]
            else:
                in_channels = self.feature_size
            crown[crown_level] = nn.Conv2d(
                in_channels,
                self.feature_size,
                kernel_size=3,
                stride=2,
                padding=1
            )
        return crown

    def forward(self, backbone_features) -> dict:
        '''
        Generate pyramid features from addition/upscale/conv with backbone features.

        Args:
            backbone_features (dict[str, torch.Tensor]): the tensors from the backbone network, keyed by
                the backbone level they came from (``BackboneNetwork.forward``).

        Returns:
            dict[str, torch.Tensor]: ``{level: (B, feature_size, H / 2**k, W / 2**k)}`` for each
            level ``Pk`` in ``levels``, where ``(H, W)`` is the backbone's input size.
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

        for i, pyr_level in enumerate(self.crown_levels):
            previous_level = f'P{level_number(pyr_level) - 1}'
            if self.legacy:
                x = pyramid_outputs[previous_level]
            elif i == 0:
                x = backbone_features[self.backbone_levels[-1]]           # P6 reads C5
            else:
                x = torch.relu(pyramid_outputs[previous_level])          # P7 and above: ReLU first
            pyramid_outputs[pyr_level] = self.crown[pyr_level](x)
        
        return pyramid_outputs

class PyramidLevel(nn.Module):

    '''
    Generates one level of the feature pyramid network that has a matched backbone level.
    '''

    def __init__(self, in_channels, feature_size=256, has_addition=False, interpolation='bilinear',
                 legacy=False):
        '''
        Args:
            in_channels (int): channels of the backbone feature at this level
            feature_size (int): channels of every pyramid level
            has_addition (bool): whether or not there is an addition step from the pyramid level above. All but the coarsest get additions.
            interpolation (str): 'bilinear' or 'nearest'
            legacy (bool): pass the smoothed output down the pyramid, as the 2026-06-30 weights expect
        '''
        super().__init__()

        self.lateral_conv = nn.Conv2d(in_channels, feature_size, kernel_size=1, stride=1)
        self.has_addition = has_addition
        self.legacy = legacy
        self.upsample = nn.Upsample(scale_factor=2, mode=interpolation)
        self.smooth_conv = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

    def forward(self, x, from_above = None):

        """
        Args:
            x (torch.Tensor): backbone feature at this level, ``(B, in_channels, h, w)``.
            from_above (torch.Tensor): ``upsampled`` from the level above, ``(B, feature_size, h, w)``,
                or ``None`` for the coarsest level.

        Returns:
            (output, upsampled): this level's pyramid feature, ``(B, feature_size, h, w)``, and the
            tensor to add into the level below,
            ``(B, feature_size, 2h, 2w)``. As in deepcell's ``fpn.py:83-126``, ``upsampled`` is the
            merged lateral, upsampled; the 3x3 conv only produces ``output``.
        """

        # Lateral connection from backbone feature to pyramid
        lateral = self.lateral_conv(x)

        # Add if we have a top-down connection
        if self.has_addition and from_above is not None:
            lateral = lateral + from_above

        # Smooth with a 3x3 conv
        output = self.smooth_conv(lateral)

        # Upsample for next level down
        upsampled = self.upsample(output if self.legacy else lateral)

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
