import os
import datetime
import json
import random
import time

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


def train_step(model, loss_fn, optimizer, image, labels, max_norm=CLIPNORM, return_heads=False):
    """One optimisation step: forward, loss, backward, per-tensor clip, Adam update.

    Returns:
        The loss (a detached scalar tensor). With ``return_heads=True``, a tuple
        ``(loss, head_losses)``, where ``head_losses`` is a detached 1-D tensor
        of the per-head losses (``loss_fn`` must then have ``head_losses``, as
        ``SemanticLoss`` does).

    Raises:
        FloatingPointError: if the loss or any gradient is NaN or inf.
    """
    optimizer.zero_grad()

    outputs = model(image)
    if return_heads:
        heads = loss_fn.head_losses(outputs, labels)
        curr_loss = sum(heads[1:], heads[0])
    else:
        curr_loss = loss_fn(outputs, labels)

    if not torch.isfinite(curr_loss):
        raise FloatingPointError(
            f'Training loss is {curr_loss.item()}. Check the batch: images should be finite, '
            'and the pixelwise targets one-hot.'
        )

    curr_loss.backward()
    clip_grad_norm_per_tensor_(model.named_parameters(), max_norm=max_norm)
    optimizer.step()

    if return_heads:
        return curr_loss.detach(), torch.stack([h.detach() for h in heads])
    return curr_loss.detach()


def head_names(n_semantic_classes, model_type='mesmer'):
    """Readable names for the loss of each head, for logs.

    Mesmer's heads are nuclear first, then whole-cell (the reverse of deepcell's order).
    """
    if model_type == 'mesmer' and list(n_semantic_classes) == [1, 3, 1, 3]:
        return ['nuc_inner_distance', 'nuc_pixelwise', 'cell_inner_distance', 'cell_pixelwise']
    return [f'head{i}' for i in range(len(n_semantic_classes))]


def _train_one_epoch(model, loss_fn, optimizer, dataloader, device, epoch):
    """One pass of ``train_step`` over the training loader.

    Returns:
        ``(loss, head_losses)``: the mean loss per sample, and a list of the
        mean loss of each head.
    """
    model.train()
    tracker = LossTracker()
    head_sums = torch.zeros(len(loss_fn.n_semantic_classes), dtype=torch.float64)

    progress = tqdm(dataloader, desc=f'Epoch {epoch} [Train]', dynamic_ncols=True, mininterval=30)
    for image, labels in progress:
        batch_size = image.shape[0]
        image = image.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        batch_loss, batch_heads = train_step(model, loss_fn, optimizer, image, labels, return_heads=True)

        tracker.update(batch_loss, batch_size=batch_size)
        head_sums += batch_heads.cpu().double() * batch_size
        progress.set_postfix({'loss': f"{tracker.get_loss():.4f}"})

    return tracker.get_loss(), (head_sums / max(tracker.total_samples, 1)).tolist()


@torch.no_grad()
def _validate(model, loss_fn, valloader, device, epoch):
    """The loss on the validation loader, with no gradient.

    Returns:
        ``(loss, head_losses, sample)``. ``loss`` and ``head_losses`` are as in
        ``_train_one_epoch``. ``sample`` is ``(targets, outputs)`` of the first
        image of the last batch, each ``(channels, H, W)``, for the TensorBoard
        figures. An empty loader leaves them unset, so this raises
        ``UnboundLocalError`` (a known issue, left as it is).
    """
    model.eval()
    tracker = LossTracker()
    head_sums = torch.zeros(len(loss_fn.n_semantic_classes), dtype=torch.float64)

    progress = tqdm(valloader, desc=f'Epoch {epoch} [Val]', dynamic_ncols=True, mininterval=30)
    for image, labels in progress:
        batch_size = image.shape[0]
        image = image.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(image)
        batch_heads = torch.stack(loss_fn.head_losses(outputs, labels))

        tracker.update(batch_heads.sum(), batch_size=batch_size)
        head_sums += batch_heads.cpu().double() * batch_size
        progress.set_postfix({'loss': f"{tracker.get_loss():.4f}"})

    sample = (labels[0], outputs[0])
    return tracker.get_loss(), (head_sums / max(tracker.total_samples, 1)).tolist(), sample


def _log_to_tensorboard(writer, epoch, names, val_loss, train_heads, val_heads, sample, model_type):
    """Write the epoch's val loss, the loss of each head and the sample figures.

    The figures show targets against outputs for the first compartment
    (channels 0-3) and, for Mesmer, the second (channels 4-7).
    """
    writer.add_scalar('avg_loss/val', val_loss, epoch)
    for name, train_head, val_head in zip(names, train_heads, val_heads):
        writer.add_scalar(f'head_loss_train/{name}', train_head, epoch)
        writer.add_scalar(f'head_loss_val/{name}', val_head, epoch)

    targets, outputs = sample
    writer.add_figure('sample_comp1', create_sample_overlay(targets[:4], outputs[:4]), epoch)
    if model_type == 'mesmer':
        writer.add_figure('sample_comp2', create_sample_overlay(targets[4:], outputs[4:]), epoch)
    writer.flush()


def _max_gpu_memory_gib(device):
    """Peak GPU memory allocated on ``device`` so far, in GiB; ``None`` off the GPU."""
    if str(device).startswith('cuda'):
        return round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
    return None


def _append_history(history_path, record):
    """Append ``record`` to ``history_path`` as one line of JSON."""
    with open(history_path, 'a') as f:
        f.write(json.dumps(record) + '\n')


def _print_epoch(names, train_loss, train_heads, val_loss, val_heads, t_train, t_epoch):
    """Print the epoch's losses (total and per head) and timings."""
    print(f'Training loss: {train_loss:.4f}  '
          + '  '.join(f'{n}={v:.4f}' for n, v in zip(names, train_heads)))
    print(f'Validation loss: {val_loss:.4f}  '
          + '  '.join(f'{n}={v:.4f}' for n, v in zip(names, val_heads)))
    print(f'Epoch time: {t_epoch:.0f} s (train {t_train:.0f} s)', flush=True)
    print()


def _save_weights(model, save_path_prefix, filename):
    """Save the model's state_dict as ``<save_path_prefix>/<filename>``."""
    torch.save(model.state_dict(), save_path_prefix + "/" + filename)


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
        model_type = 'mesmer',
        history_path=None,
    ):
    """Train with the published recipe; keep the checkpoint with the lowest val loss.

    Each epoch trains on every batch of ``dataloader``, measures the loss on
    ``valloader``, then lowers the learning rate (``make_optimizer``). There is
    no early stopping.

    Args:
        dataloader, valloader: loaders of ``(images, targets)`` batches, as
            ``create_data_loaders`` returns.
        model: the network to train (required), already on ``device``. It
            must have ``n_semantic_classes``, as ``PanopticNet`` does.
        lr: initial learning rate.
        epochs: number of epochs.
        save_path_prefix: folder for the checkpoints.
        writer: optional TensorBoard ``SummaryWriter``. Gets the lr, the losses
            and sample figures every epoch.
        write: if ``False``, save no checkpoints.
        device: where to move each batch.
        ce_weight: weight of the cross-entropy heads (``SemanticLoss``).
        model_type: ``'mesmer'`` gives the heads readable names in the logs
            and draws figures for both compartments.
        history_path: optional JSON-lines file, see below.

    Returns:
        ``model``, trained in place.

    Side effects:
        Saves the state_dict as ``saved_model_best_dict.pth`` (whenever the val
        loss is the lowest so far) and ``last_model_dict.pth`` (at the end)
        under ``save_path_prefix``. If ``history_path`` is given, appends one
        JSON line per epoch to it: lr, train and val loss (total and per
        head), whether it was the best so far, and timings. Prints a summary
        of each epoch.
    """
    assert model is not None, "Please specify a model"

    names = head_names(model.n_semantic_classes, model_type)
    loss_fn = SemanticLoss(n_semantic_classes=model.n_semantic_classes, ce_weight=ce_weight)
    optimizer, scheduler = make_optimizer(model, lr=lr)
    best_val_loss = None

    for epoch in range(epochs):
        epoch_lr = optimizer.param_groups[0]['lr']
        if writer is not None:
            writer.add_scalar('lr', epoch_lr, epoch)

        t_start = time.time()
        train_loss, train_heads = _train_one_epoch(model, loss_fn, optimizer, dataloader, device, epoch)
        t_train = time.time() - t_start
        if writer is not None:
            writer.add_scalar('avg_loss/train', train_loss, epoch)

        val_loss, val_heads, sample = _validate(model, loss_fn, valloader, device, epoch)
        t_epoch = time.time() - t_start

        if writer is not None:
            _log_to_tensorboard(writer, epoch, names, val_loss, train_heads, val_heads, sample, model_type)

        # Per-epoch exponential decay; no ReduceLROnPlateau, no early stopping.
        scheduler.step()

        is_best = best_val_loss is None or val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            if write:
                _save_weights(model, save_path_prefix, "saved_model_best_dict.pth")
            print("\nNew best model.\n")

        if history_path is not None:
            _append_history(history_path, {
                'epoch': epoch,
                'lr': epoch_lr,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_head_loss': dict(zip(names, train_heads)),
                'val_head_loss': dict(zip(names, val_heads)),
                'is_best': is_best,
                'train_seconds': round(t_train, 1),
                'epoch_seconds': round(t_epoch, 1),
                'max_gpu_memory_gib': _max_gpu_memory_gib(device),
            })

        _print_epoch(names, train_loss, train_heads, val_loss, val_heads, t_train, t_epoch)

    if write:
        _save_weights(model, save_path_prefix, "last_model_dict.pth")

    return model

def _training_config():
    """The settings of a training run: the published recipe, plus paths, device and worker count."""
    return {
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


def _make_run_dirs(config, start_time):
    """Create ``<run_info>/<start_time>`` and, if checkpoints are saved,
    ``<model_path>/<start_time>``.

    Returns:
        ``(run_info, model_path)``, the two folders as strings.
    """
    run_info = config['run_info'] + '/' + start_time
    model_path = config['model_path'] + '/' + start_time

    if not os.path.isdir(run_info):
        os.makedirs(run_info, exist_ok=True)
    if not os.path.isdir(model_path) and config['write']:
        os.makedirs(model_path, exist_ok=True)

    return run_info, model_path


def _build_model(config):
    """A new ``PanopticNet`` for ``config``, on ``config['device']``."""
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
    return model


def _build_loaders(config, z_train, z_val):
    """The train and val loaders for ``config`` (see ``create_data_loaders``)."""
    return create_data_loaders(
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


def main():
    """Train Mesmer on TissueNet v1.1 with the published recipe.

    Reads ``train.zarr`` and ``val.zarr`` from ``~/.deepcell/tissuenet_v1-1/``.
    Each run gets a timestamped folder under ``data/logs/`` (TensorBoard logs,
    ``history.jsonl``) and one under ``data/model/`` (the checkpoints, see
    ``train_torch``).
    """
    # Set here rather than at import time, so importing train.py (from tests or
    # other scripts) leaves the thread pool alone.
    torch.set_num_interop_threads(24)

    config = _training_config()
    seed_everything(config['seed'])
    start_time = f"{datetime.datetime.now():%Y%m%d%H%M%S}"

    z_train = zarr.open(f"{config['data_path']}/train.zarr", mode="r")
    z_val = zarr.open(f"{config['data_path']}/val.zarr", mode="r")

    run_info, model_path = _make_run_dirs(config, start_time)
    writer = SummaryWriter(run_info)

    model = _build_model(config)
    train_data, val_data = _build_loaders(config, z_train, z_val)

    train_torch(
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
        model_type = config['model_type'],
        history_path=f'{run_info}/history.jsonl',
    )

    writer.close()

if __name__ == "__main__":
    main()
