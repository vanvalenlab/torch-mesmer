import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import numpy as np
 
from torch_mesmer.model_utils import create_model
from torch_mesmer.file_utils import _load_npz, load_data

from torch_mesmer.loader_utils import SemanticDataset
from torch_mesmer.loader_utils import CroppingDatasetTorch

from torch.utils.data import Dataset, DataLoader


from torch_mesmer.mesmer import mesmer_preprocess


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    tissuenet_dir = "/data/tissuenet"
    (X_train, y_train), (X_val, y_val) = load_data(tissuenet_dir)

    crop_size = 256
    backbone = 'resnet50'
    lr = 0.0001
    # create model and move it to GPU with id rank
    model, losses, optimizer = create_model(
        input_shape=(crop_size, crop_size, 2),
        backbone=backbone,
        lr=lr,
        device=rank,
    ) 
    dummy = torch.rand(3, 2, crop_size, crop_size).to(rank)
    model = model.to(rank)
    model(dummy)
    
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=lr)

    smaller = 1024
    smaller_test = None
    if smaller:
        X_train, y_train = X_train[:smaller], y_train[:smaller]
        X_val, y_val = X_val[:smaller], y_val[:smaller]
    if smaller_test:
        X_test, y_test = X_test[:smaller_test], y_test[:smaller_test]

    # partition = len(X_train)//world_size

    # if rank == (world_size-1):
    #     X_train = X_train[partition*rank:]
    #     y_train = y_train[partition*rank:]
    # else:
    #     X_train = X_train[partition*rank:partition*(rank+1)]
    #     y_train = y_train[partition*rank:partition*(rank+1)]

    X_train = mesmer_preprocess(X_train)
    X_val = mesmer_preprocess(X_val)

    seed = 0
    zoom_min = 0.75
    batch_size = 8

    rotation_range = 180
    shear_range = 0
    zoom_range = (zoom_min, 1/zoom_min)
    horizontal_flip = True
    vertical_flip = True
    
    transforms=["inner-distance", "pixelwise"]
    transforms_kwargs={
        "pixelwise": {"dilation_radius": 1},
        "inner-distance": {"erosion_width": 1, "alpha": "auto"},
    }
    
    cdt = CroppingDatasetTorch(X_train, y_train, rotation_range, shear_range, zoom_range, horizontal_flip, vertical_flip, crop_size, batch_size=batch_size, transforms=transforms, transforms_kwargs=transforms_kwargs)
    dataloader = DataLoader(cdt, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, sampler=DistributedSampler(cdt))
    sd = SemanticDataset(X_val, y_val, transforms=transforms, transforms_kwargs=transforms_kwargs)
    valloader = DataLoader(sd, batch_size=batch_size, shuffle=False, num_workers=4)

    def train_one_epoch(model):
        running_loss_avg = 0.
        count = 0
    
        for (li_inputs, li_labels) in tqdm(dataloader):
            count += 1
                    
            inputs = li_inputs.to(rank)
            labels = [l.to(rank) for l in li_labels]
    
            optimizer.zero_grad()
    
            outputs = model(inputs)
    
            loss = sum([losses[j](outputs[j], labels[j]) for j in range(len(losses))])
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.001, error_if_nonfinite=True)
        
            optimizer.step()
    
            running_loss_avg += loss.item()
            
        return running_loss_avg/count    

    loss_tracking = []
    vloss_tracking = []
    decay_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.33, patience=5,)

    epoch_number = 0
    start_epoch = 0
    EPOCHS = 5

    best_vloss = 1_000_000.
    patience_count = 0

    save_path_prefix = "/data/saved_model"
    if smaller is None:
        save_path_prefix = save_path_prefix + "_ddp_full_" + str(batch_size)
    else:
        save_path_prefix = save_path_prefix + "_ddp_" + str(smaller)


    for epoch in range(start_epoch, EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
        print("TRAIN")
        ddp_model.train(True)
        avg_loss = train_one_epoch(ddp_model)
        count = 0
        
        running_vloss_avg = 0.
        
        print("VAL")
        ddp_model.eval()

        with torch.no_grad():
            for (li_inputs, li_labels) in tqdm(valloader):
                count += 1
                
                vinputs = li_inputs.to(rank)
                vlabels = [l.to(rank) for l in li_labels]
                
                voutputs = ddp_model(vinputs)
                
                vloss = sum([losses[j](voutputs[j], vlabels[j]) for j in range(len(losses))])
                    
                running_vloss_avg += vloss
                    
        avg_vloss = running_vloss_avg/count

        decay_scheduler.step()
        plateau_scheduler.step(avg_vloss)
        if rank == 0:
            print(decay_scheduler.get_last_lr())
        
        loss_tracking.append(avg_loss)
        vloss_tracking.append(avg_vloss)
        
        # Save model periodically
        # TODO: figure out if you should load all as well
        if rank == 0 and (epoch+1)%10==0:
            epoch_path_prefix = save_path_prefix + "_epoch" + str(epoch+1)
            dict_save_path = epoch_path_prefix + "_dict.pth"            
            torch.save(ddp_model.module.state_dict(), dict_save_path)
        
        if rank == 0 and avg_vloss<best_vloss:
            best_vloss = avg_vloss
            dict_save_path = save_path_prefix + "_best_dict.pth"            
            torch.save(ddp_model.module.state_dict(), dict_save_path)

            patience_count = 0
        elif rank == 0:
            patience_count += 1

        if rank == 0: 
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        epoch_number += 1

        if patience_count >= 10:
            break

    cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
if __name__ == '__main__':
    run_demo(demo_basic, 2)