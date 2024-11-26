from .iter_semantic import SemanticDataGenerator
from .iter_cropping import CroppingDataGenerator

def create_data_generators(
    train_dict,
    val_dict,
    rotation_range=180,
    shear_range=0,
    zoom_min=0.7,
    horizontal_flip=True,
    vertical_flip=True,
    crop_size=(256, 256),
    seed=0,
    batch_size=8,
    min_objects=0,
    data_format='channels_last',
):
    # use augmentation for training but not validation
    datagen = CroppingDataGenerator(
        rotation_range=rotation_range,
        shear_range=shear_range,
        zoom_range=(zoom_min, 1 / zoom_min),
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        crop_size=(crop_size, crop_size),
        data_format=data_format,
    )

    datagen_val = SemanticDataGenerator(
        rotation_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=0,
        vertical_flip=0,
        data_format=data_format,
    )

    train_data = datagen.flow(
        train_dict,
        seed=seed,
        transforms=["inner-distance", "pixelwise"],
        transforms_kwargs={
            "pixelwise": {"dilation_radius": 1},
            "inner-distance": {"erosion_width": 1, "alpha": "auto"},
        },
        min_objects=0,
        batch_size=batch_size,
    )

    val_data = datagen_val.flow(
        val_dict,
        seed=seed,
        transforms=["inner-distance", "pixelwise"],
        transforms_kwargs={
            "pixelwise": {"dilation_radius": 1},
            "inner-distance": {"erosion_width": 1, "alpha": "auto"},
        },
        min_objects=min_objects,
        batch_size=batch_size,
    )

    return train_data, val_data