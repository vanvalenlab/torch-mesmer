import os
import datetime
import random

import numpy as np
import torch
import zarr

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

from torch_mesmer.model import PanopticNet
from torch_mesmer.loss import SemanticLoss, LossTracker
from torch_mesmer.loaders import (CLAHE_KERNEL_SIZE, IN_TRANSFORMS, MIN_OBJECT_SIZE, MIN_OBJECTS,
                                   TRANSFORMS_KWARGS, ZOOM_MIN, create_data_loaders)
from torch_mesmer.utils import create_sample_overlay

# Optimiser settings of the published Mesmer recipe.
ADAM_EPSILON = 1e-7   # tf.keras Adam default; PyTorch's default is 1e-8
CLIPNORM = 0.001      # Keras Adam(clipnorm=0.001): applied to each gradient tensor separately
LR_DECAY = 0.99       # lr = lr0 * 0.99**epoch (deepcell rate_scheduler)


def seed_everything(seed=0):
    """Seed Python, NumPy and PyTorch (CPU and every GPU).

    DataLoader workers get their own seeds from the loader's generator (see
    ``create_data_loaders``), so this plus that generator makes a run repeatable
    on the same hardware.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_optimizer(model, lr=1e-4):
    """Adam with the Keras epsilon, plus the per-epoch exponential LR decay.

    Call ``scheduler.step()`` once at the end of every epoch, so epoch ``e``
    (counting from 0) trains at ``lr * LR_DECAY**e``.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=ADAM_EPSILON)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: LR_DECAY ** epoch)
    return optimizer, scheduler


@torch.no_grad()
def clip_grad_norm_per_tensor_(named_parameters, max_norm=CLIPNORM):
    """Clip each parameter's gradient to norm ``max_norm`` on its own.

    This is Keras ``clipnorm`` (``tf.clip_by_norm`` on every gradient):
    ``g <- g * max_norm / max(||g||, max_norm)``. It is not
    ``torch.nn.utils.clip_grad_norm_``, which scales all gradients by one
    global norm (Keras's separate ``global_clipnorm``).

    Args:
        named_parameters: iterable of (name, parameter), e.g. ``model.named_parameters()``.
        max_norm: the per-tensor norm limit.

    Returns:
        Tensor of the gradient norms before clipping, one per parameter that has a gradient.

    Raises:
        FloatingPointError: if any gradient contains NaN or inf. The message names the
        affected parameters.
    """
    names, grads = [], []
    for name, p in named_parameters:
        if p.grad is not None:
            names.append(name)
            grads.append(p.grad)

    if not grads:
        return torch.zeros(0)

    norms = torch.stack([torch.linalg.vector_norm(g.float()) for g in grads])

    finite = torch.isfinite(norms)
    if not finite.all():
        bad = [names[i] for i in torch.nonzero(~finite).flatten().tolist()]
        shown = ', '.join(bad[:5]) + (f', ... ({len(bad)} in total)' if len(bad) > 5 else '')
        raise FloatingPointError(
            f'Non-finite gradient in {len(bad)} of {len(names)} parameter tensors: {shown}. '
            'If the loss was finite, the problem is inside the backward pass '
            '(e.g. an overflow under mixed precision, or a bad input image).'
        )

    for g, norm in zip(grads, norms):
        g.mul_((max_norm / torch.clamp(norm, min=max_norm)).to(g.dtype))

    return norms


def train_step(model, loss_fn, optimizer, image, labels, max_norm=CLIPNORM):
    """One optimisation step: forward, loss, backward, per-tensor clip, Adam update.

    Returns:
        The loss (a detached scalar tensor).

    Raises:
        FloatingPointError: if the loss or any gradient is NaN or inf.
    """
    optimizer.zero_grad()

    outputs = model(image)
    curr_loss = loss_fn(outputs, labels)

    if not torch.isfinite(curr_loss):
        raise FloatingPointError(
            f'Training loss is {curr_loss.item()}. Check the batch: images should be finite, '
            'and the pixelwise targets one-hot.'
        )

    curr_loss.backward()
    clip_grad_norm_per_tensor_(model.named_parameters(), max_norm=max_norm)
    optimizer.step()

    return curr_loss.detach()


def train_torch(
        dataloader,
        valloader,
        model=None,
        lr=1e-4,
        epochs=100,
        save_path_prefix = "data/saved_model",
        writer=None,
        write=True,
        device='cuda',
        ce_weight=0.01,
        model_type = 'mesmer'
    ):

    assert model is not None, "Please specify a model"

    n_semantic_classes = model.n_semantic_classes

    loss = SemanticLoss(n_semantic_classes=n_semantic_classes, ce_weight=ce_weight)
    optimizer, scheduler = make_optimizer(model, lr=lr)

    for epoch in range(epochs):
        if writer is not None:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = LossTracker()
        val_loss = LossTracker()
        model.train()

        pbar_train = tqdm(dataloader, desc=f'Epoch {epoch} [Train]', dynamic_ncols=True)

        for batch in pbar_train:
            
            image, labels = batch
            batch_size = image.shape[0]
            
            image = image.to(device)
            labels = labels.to(device)

            curr_loss = train_step(model, loss, optimizer, image, labels)

            train_loss.update(curr_loss, batch_size=batch_size)

            pbar_train.set_postfix({
                'loss': f"{train_loss.get_loss():.4f}"
            })

        if writer is not None:
            writer.add_scalar('avg_loss/train', train_loss.get_loss(), epoch)

        model.eval()
        pbar_val = tqdm(valloader, desc=f'Epoch {epoch} [Val]', dynamic_ncols=True)

        with torch.no_grad():
            for batch in pbar_val:
                image, labels = batch

                batch_size = image.shape[0]
                
                image = image.to(device)
                labels = labels.to(device)

                voutputs = model(image)
                curr_vloss = loss(voutputs, labels)

                val_loss.update(curr_vloss, batch_size=batch_size)  

                pbar_val.set_postfix({
                    'loss': f"{val_loss.get_loss():.4f}"
                })

        # Shape 8, H, W
        sampled_label = labels[0]
        sampled_transforms = voutputs[0]

        avg_vloss = val_loss.get_loss()
        if writer is not None:
            writer.add_scalar('avg_loss/val', avg_vloss, epoch)

            c1_figure = create_sample_overlay(sampled_label[:4], sampled_transforms[:4])
            writer.add_figure('sample_comp1', c1_figure, epoch)

            if model_type == 'mesmer':
                c2_figure = create_sample_overlay(sampled_label[4:], sampled_transforms[4:])
                writer.add_figure('sample_comp2', c2_figure, epoch)

        # Per-epoch exponential decay; no ReduceLROnPlateau, no early stopping.
        scheduler.step()

        if epoch == 0:

            best_vloss = avg_vloss
            
            if write:
                dict_save_path = save_path_prefix + "/saved_model_best_dict.pth"
                torch.save(model.state_dict(), dict_save_path)

            print()
            print("New best model.")
            print()  

        elif avg_vloss < best_vloss:
            best_vloss = avg_vloss
            
            if write:
                dict_save_path = save_path_prefix + "/saved_model_best_dict.pth"
                torch.save(model.state_dict(), dict_save_path)

            print()
            print("New best model.")
            print()
            
        print(f'Training loss: {train_loss.get_loss():.3f}')
        print(f'Validation loss: {val_loss.get_loss():.3f}')
        print()

    if write:
        dict_save_path = save_path_prefix + "/last_model_dict.pth"
        torch.save(model.state_dict(), dict_save_path)

    return model

def main():

    # Set here rather than at import time, so importing train.py (from tests or
    # other scripts) leaves the thread pool alone.
    torch.set_num_interop_threads(24)

    config = {
        'model_path': "data/model/",
        'data_path': Path.home() / '.deepcell/tissuenet_v1-1/',
        'run_info': 'data/logs/',
        'epochs': 100,
        'batch_size': 8,
        'backbone': 'resnet50',
        'lr': 1e-4,
        # Data pipeline of the published recipe; passed to create_data_loaders.
        'crop_size': 256,
        'zoom_min': ZOOM_MIN,
        'rotation_range': 180,
        'in_transforms': list(IN_TRANSFORMS),
        'transforms_kwargs': TRANSFORMS_KWARGS,
        'kernel_size': CLAHE_KERNEL_SIZE,
        'min_object_size': MIN_OBJECT_SIZE,   # train only
        'min_objects': MIN_OBJECTS,           # train only
        'pyramid_levels': ['P3', 'P4', 'P5', 'P6', 'P7'],
        'backbone_levels': ['C3', 'C4', 'C5'],
        'num_workers': 16,
        'write': True,
        'device': 'cuda',
        'n_semantic_classes': [1,3,1,3],
        'ce_weight': 0.01,
        'seed': 0,
        'model_type': 'mesmer',
        'notes': 'removed lazy conv in favor of defined in and out features'
    }

    seed_everything(config['seed'])
    curr_time = f"{datetime.datetime.now():%Y%m%d%H%M%S}"

    z_train = zarr.open(f"{config['data_path']}/train.zarr", mode="r")
    z_val = zarr.open(f"{config['data_path']}/val.zarr", mode="r")

    run_info = config['run_info'] + '/' + curr_time
    model_path = config['model_path'] + '/' + curr_time
    
    if not os.path.isdir(run_info):
        os.makedirs(run_info, exist_ok=True)
    if not os.path.isdir(model_path) and config['write']:
        os.makedirs(model_path, exist_ok=True)

    writer = SummaryWriter(run_info)
    
    print("Initializing model:")
    print()

    model = PanopticNet(
        crop_size=config['crop_size'],
        backbone=config['backbone'],
        pyramid_levels=config['pyramid_levels'],
        backbone_levels=config['backbone_levels'],
        n_semantic_classes = config['n_semantic_classes']
    )

    model = model.to(config['device'])

    print("Panoptic Model:")
    print(f"    Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Set up data generators with updated data
    train_data, val_data = create_data_loaders(
        z_train,
        z_val,
        crop_size=config['crop_size'],
        zoom_min=config['zoom_min'],
        rotation_range=config['rotation_range'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        in_transforms=config['in_transforms'],
        transforms_kwargs=config['transforms_kwargs'],
        kernel_size=config['kernel_size'],
        min_object_size=config['min_object_size'],
        min_objects=config['min_objects'],
        semantic_heads=config['n_semantic_classes'],
        seed=config['seed'],
    )

    # train the model
    model = train_torch(
        train_data,
        val_data,
        model=model,
        lr=config['lr'],
        epochs=config['epochs'],
        save_path_prefix=model_path,
        writer=writer,
        write=config['write'],
        ce_weight=config['ce_weight'],
        device=config['device'],
        model_type = config['model_type']
    )

    writer.close()

if __name__ == "__main__":
    main()
