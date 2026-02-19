import torch
torch.set_num_threads(24)
import os
import datetime
import zarr

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from torch_mesmer.model import PanopticNet
from torch_mesmer.loss import SemanticLoss, LossTracker
from torch_mesmer.loaders import create_data_loaders
from torch_mesmer.utils import create_sample_overlay

def train_torch(
        dataloader,
        valloader,
        model=None,
        lr=1e-4,
        epochs=8,
        save_path_prefix = "data/saved_model",
        writer=None,
        write=True,
        device='cuda:2',
        loss_weight=0.01,
        model_type = 'mesmer'
    ):

    assert model is not None, "Please specify a model"

    n_semantic_classes = model.n_semantic_classes

    loss = SemanticLoss(n_semantic_classes=n_semantic_classes, loss_weight=loss_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.33, patience=5,)

    for epoch in range(epochs):

        train_loss = LossTracker()
        val_loss = LossTracker()
        model.train()

        pbar_train = tqdm(dataloader, desc=f'Epoch {epoch} [Train]', dynamic_ncols=True)

        for batch in pbar_train:
            
            image, labels = batch
            batch_size = image.shape[0]
            
            image = image.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(image)
            curr_loss = loss(outputs, labels) 

            train_loss.update(curr_loss, batch_size=batch_size)  

            pbar_train.set_postfix({
                'loss': f"{train_loss.get_loss():.4f}"
            })

            curr_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.001, error_if_nonfinite=True)
        
            optimizer.step()

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
        writer.add_scalar('avg_loss/val', avg_vloss, epoch)

        c1_figure = create_sample_overlay(sampled_label[:4], sampled_transforms[:4])
        writer.add_figure('sample_comp1', c1_figure, epoch)
        
        if model_type == 'mesmer':
            c2_figure = create_sample_overlay(sampled_label[4:], sampled_transforms[4:])
            writer.add_figure('sample_comp2', c2_figure, epoch)

        plateau_scheduler.step(avg_vloss)

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

    config = {
        'model_path': "data/model/",
        'data_path': '/data/shared/caliban/DynamicNuclearNet-segmentation-v1_0',
        'run_info': 'data/logs/',
        'epochs': 16,
        'zoom_min': 0.75,
        'batch_size': 4,
        'backbone': 'resnet50',
        'crop_size': 256,
        'lr': 1e-4,
        'outer_erosion_width': 1,
        'inner_distance_alpha': 'auto',
        'inner_distance_beta': 1,
        'inner_erosion_width': 0,
        'pyramid_levels': ['P3', 'P4', 'P5', 'P6', 'P7'],
        'backbone_levels': ['C3', 'C4', 'C5'],
        'num_workers': 8,
        'write': True,
        'device': 'cuda:1',
        'n_semantic_classes': [1,3],
        'loss_weight': 0.01,
        'model_type': 'caliban'
    }

    curr_time = f"{datetime.datetime.now():%Y%m%d%H%M%S}"

    z_train = zarr.open(f"{config['data_path']}/train.zarr")
    z_val = zarr.open(f"{config['data_path']}/val.zarr")

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

    # Dummy data for initializing the lazyconv sizes
    dummy_data = torch.rand(1, 2, config['crop_size'], config['crop_size']).to(config['device'])
    _ = model(dummy_data)
    del dummy_data

    print("Panoptic Model:")
    print(f"    Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Set up data generators with updated data
    train_data, val_data = create_data_loaders(
        z_train,
        z_val,
        crop_size=config['crop_size'],
        zoom_min=config['zoom_min'],
        batch_size=config['batch_size'],
        data_format='channels_last',
        num_workers=config['num_workers'],
        semantic_heads=config['n_semantic_classes']
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
        loss_weight=config['loss_weight'],
        device=config['device'],
        model_type = config['model_type']
    )

    writer.close()

if __name__ == "__main__":
    main()
