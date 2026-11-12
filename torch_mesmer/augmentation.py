"""Training augmentation for the Mesmer data pipeline.

One ``MultiTransform`` is drawn per training sample and applied to the image
and every target channel, so they all move together.

Order (this port's own, kept on purpose; the published recipe crops to
``crop_size`` first and then applies one affine transform):

1. Take a random crop of ``crop_size * zy`` rows by ``crop_size * zx`` columns,
   with ``zx`` and ``zy`` drawn independently from ``(zoom_min, 1 / zoom_min)``
   (the published recipe's anisotropic zoom range, 0.7 to 1.43).
2. Resize the crop to ``crop_size x crop_size``.
3. Rotate by an angle drawn from ``U(-rotation_range, rotation_range)``.
4. Flip left-right and up-down, each with probability 0.5.

Steps 2 and 3 are done in a single resampling, as the recipe's Keras affine is,
so the image is interpolated once rather than twice. The corners that rotation
exposes are filled with the nearest edge pixel of the crop (Keras
``fill_mode='nearest'``), and nothing is antialiased, as in Keras. Images and
distance maps use bilinear interpolation; one-hot labels use nearest, which
picks the same source pixel for every class channel, so they stay one-hot.

Validation samples are not augmented. They are only centre-cropped (or
edge-padded) to ``crop_size`` when their size differs from it.
"""

import math
import random

import torch
import torch.nn.functional as F

INTERPOLATION_MODES = ('bilinear', 'nearest')


class MultiTransform:
    """Random geometric augmentation shared by an image and its targets.

    Args:
        image_shape: ``(H, W)`` of the sample to be transformed.
        dataset_type: ``'train'`` draws random augmentation parameters;
            anything else applies no augmentation.
        crop_size: side of the square output.
        rotation_range: rotation angle limit in degrees.
        zoom_min: the zoom factors are drawn from ``U(zoom_min, 1/zoom_min)``.
            A zoom ``z`` crops ``crop_size * z`` pixels along that axis, so
            ``z > 1`` zooms out.
    """

    def __init__(self, image_shape, dataset_type='val', crop_size=256, rotation_range=180, zoom_min=0.7):
        if not 0 < zoom_min <= 1:
            raise ValueError(f'zoom_min must be in (0, 1], got {zoom_min}')

        self.image_shape = tuple(int(s) for s in image_shape)
        self.dataset_type = dataset_type
        self.crop_size = int(crop_size)
        self.rotation_range = rotation_range
        self.zoom_min = zoom_min
        self.augment = dataset_type == 'train'
        self._draw_params()

    def _draw_params(self):
        """Set the crop box, angle and flips: random for training, a plain
        centre crop otherwise."""
        H, W = self.image_shape

        if self.augment:
            # Same draws as Keras: zx, zy independent (keras-preprocessing
            # image_data_generator.py:797-803), then the angle, then the flips.
            zoom_y = random.uniform(self.zoom_min, 1 / self.zoom_min)
            zoom_x = random.uniform(self.zoom_min, 1 / self.zoom_min)
            self.crop_h = max(1, round(self.crop_size * zoom_y))
            self.crop_w = max(1, round(self.crop_size * zoom_x))
            self.top = _random_offset(H, self.crop_h)
            self.left = _random_offset(W, self.crop_w)
            self.angle = random.uniform(-self.rotation_range, self.rotation_range)
            self.do_hflip = random.random() < 0.5
            self.do_vflip = random.random() < 0.5
        else:
            self.crop_h = self.crop_w = self.crop_size
            self.top = (H - self.crop_size) // 2
            self.left = (W - self.crop_size) // 2
            self.angle = 0.0
            self.do_hflip = self.do_vflip = False

    def __call__(self, image, interpolation):
        """Transform a ``(C, H, W)`` tensor.

        Args:
            image: float tensor, ``(C, H, W)``, with ``(H, W) == image_shape``.
            interpolation: one mode per channel, each ``'bilinear'`` or ``'nearest'``.

        Returns:
            float tensor, ``(C, crop_size, crop_size)``.
        """
        if image.ndim != 3:
            raise ValueError(f'expected a (C, H, W) tensor, got shape {tuple(image.shape)}')
        if tuple(image.shape[1:]) != self.image_shape:
            raise ValueError(f'image is {tuple(image.shape[1:])}, but the transform was drawn for {self.image_shape}')
        if len(interpolation) != image.shape[0]:
            raise ValueError(f'{len(interpolation)} interpolation modes for {image.shape[0]} channels')
        bad = set(interpolation) - set(INTERPOLATION_MODES)
        if bad:
            raise ValueError(f'unknown interpolation mode(s) {sorted(bad)}; use {INTERPOLATION_MODES}')

        crop = self._crop(image)

        if not self.augment:
            return crop

        out = torch.empty((image.shape[0], self.crop_size, self.crop_size), dtype=torch.float32)
        grid = self._sampling_grid()
        for mode in INTERPOLATION_MODES:
            idx = [c for c, m in enumerate(interpolation) if m == mode]
            if idx:
                out[idx] = F.grid_sample(crop[idx].float().unsqueeze(0), grid, mode=mode,
                                         padding_mode='border', align_corners=False)[0]

        if self.do_hflip:
            out = out.flip(-1)
        if self.do_vflip:
            out = out.flip(-2)

        return out

    def _crop(self, image):
        """Crop ``crop_h x crop_w`` at (top, left). Rows and columns that fall
        outside the image repeat its edge pixels."""
        H, W = self.image_shape
        rows = torch.arange(self.top, self.top + self.crop_h).clamp_(0, H - 1)
        cols = torch.arange(self.left, self.left + self.crop_w).clamp_(0, W - 1)
        return image[:, rows][:, :, cols]

    def _sampling_grid(self):
        """``grid_sample`` grid that resizes the crop to the output size and
        rotates it about its centre (counter-clockwise for a positive angle, as
        torchvision's ``rotate``).

        With ``align_corners=False``, normalised coordinates of a square output
        and of the (h x w) crop differ only by the resize, so the rotation is a
        plain rotation in normalised coordinates. The output's border pixels are
        clamped to the crop's border pixels, i.e. nearest-edge fill.
        """
        theta = math.radians(self.angle)
        c, s = math.cos(theta), math.sin(theta)
        # Maps output (x, y) to source (x, y); y points down, so this inverse
        # rotation turns the content counter-clockwise on screen.
        affine = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], dtype=torch.float32).unsqueeze(0)
        return F.affine_grid(affine, (1, 1, self.crop_size, self.crop_size), align_corners=False)


def _random_offset(size, crop):
    """Uniform crop offset. When the crop is larger than the image, the image
    sits at a uniform position inside the crop instead (the rest is edge fill)."""
    if crop <= size:
        return random.randint(0, size - crop)
    return -random.randint(0, crop - size)
