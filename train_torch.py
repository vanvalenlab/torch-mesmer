import os
import numpy as np
import torch
import torch.nn as nn
import time

from tqdm import tqdm

from mesmer import mesmer_preprocess
from iter_semantic import SemanticDataGenerator
from iter_cropping import CroppingDataGenerator

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

tissuenet_dir = "/data/tissuenet"
if not os.path.exists(tissuenet_dir):
    print("Created tissuenet data dir")
    os.makedirs(tissuenet_dir)

print(os.listdir(tissuenet_dir))

from file_utils import _load_npz, load_data
from model_utils import create_model

# instantiate model
crop_size = 256
backbone = 'resnet50'
lr = 0.0001
model, losses, optimizer = create_model(
    input_shape=(crop_size, crop_size, 2),
    backbone=backbone,
    lr=lr,
    device=device,
) 
dummy = torch.rand(3, 2, crop_size, crop_size).to(device)
model(dummy)

from train_utils import create_data_generators
from loader_utils import SemanticDataset
from loader_utils import CroppingDatasetTorch

seed = 0
zoom_min = 0.75
batch_size = 8

(X_train, y_train), (X_val, y_val) = load_data(tissuenet_dir)

smaller = None
if smaller:
    X_train, y_train = X_train[:smaller], y_train[:smaller]
    X_val, y_val = X_val[:smaller], y_val[:smaller]

# train_dict = {"X": mesmer_preprocess(X_train), "y": y_train}
# val_dict = {"X": mesmer_preprocess(X_val), "y": y_val}

X_train = mesmer_preprocess(X_train)
print("train preprocess: done")

X_val = mesmer_preprocess(X_val)
print("val preprocess: done")

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
from torch.utils.data import Dataset, DataLoader

cdt = CroppingDatasetTorch(X_train, y_train, rotation_range, shear_range, zoom_range, horizontal_flip, vertical_flip, crop_size, batch_size=batch_size, transforms=transforms, transforms_kwargs=transforms_kwargs)

dataloader = DataLoader(cdt, batch_size=batch_size, shuffle=True, num_workers=4)
dataiter = iter(dataloader)
X_sd, y_sd = next(dataiter)
print(X_sd.shape)
print(len(y_sd))
print(y_sd[3].shape)


X_train = np.transpose(X_train, (0, 3, 1, 2))
y_train = np.transpose(y_train, (0, 3, 1, 2))
X_val = np.transpose(X_val, (0, 3, 1, 2))
y_val = np.transpose(y_val, (0, 3, 1, 2))
sd = SemanticDataset(X_val, y_val, transforms=transforms, transforms_kwargs=transforms_kwargs)
valloader = DataLoader(sd, batch_size=batch_size, shuffle=False, num_workers=4)


train_dict = {"X": X_train, "y": y_train}
val_dict = {"X": X_val, "y": y_val}

train_data, val_data = create_data_generators(
    train_dict,
    val_dict,
    seed=seed,
    zoom_min=zoom_min,
    batch_size=batch_size,
    crop_size=crop_size,
    data_format="channels_first"
)

inputs, labels = train_data.next()
print(inputs.shape)
print(len(labels))
print(labels[3].shape)

def train_one_epoch(model):
    running_loss_avg = 0.
    count = 0

    per_epoch_steps = len(X_train)//batch_size if len(X_train)%batch_size==0 else (len(X_train)//batch_size + 1)
    # dataloader = DataLoader(cdt, batch_size=batch_size, shuffle=True, num_workers=4)
    for (li_inputs, li_labels) in tqdm(dataloader):

    # for _ in tqdm(range(per_epoch_steps)):
        count += 1
        
        # li_inputs, li_labels = next(dataiter)
        
        inputs = li_inputs.to(device)
        labels = [l.to(device) for l in li_labels]

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
EPOCHS = 100

best_vloss = 1_000_000.
patience_count = 0

model = model.to(device)

save_path_prefix = "/data/saved_model"
if smaller is None:
    save_path_prefix = save_path_prefix + "_full_torch_" + str(batch_size)
else:
    save_path_prefix = save_path_prefix + "_" + str(smaller)

for epoch in range(start_epoch, EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))
    print("TRAIN")
    model.train(True)
    avg_loss = train_one_epoch(model)
    count = 0
    
    running_vloss_avg = 0.
    
    print("VAL")
    model.eval()

    per_epoch_steps = len(X_val)//batch_size if len(X_val)%batch_size==0 else (len(X_val)//batch_size + 1)
    with torch.no_grad():
        for (li_inputs, li_labels) in tqdm(valloader):
        # for _ in tqdm(range(per_epoch_steps)):
            count += 1
            
            # li_inputs, li_labels = val_data.next()

            vinputs = li_inputs.to(device)
            vlabels = [l.to(device) for l in li_labels]
            
            voutputs = model(vinputs)
            
            vloss = sum([losses[j](voutputs[j], vlabels[j]) for j in range(len(losses))])
                
            running_vloss_avg += vloss
                
    avg_vloss = running_vloss_avg/count

    decay_scheduler.step()
    plateau_scheduler.step(avg_vloss)
    print(decay_scheduler.get_last_lr())
    
    loss_tracking.append(avg_loss)
    vloss_tracking.append(avg_vloss)
    
    # Save model periodically
    if (epoch+1)%10==0:
        epoch_path_prefix = save_path_prefix + "_epoch" + str(epoch+1)
        dict_save_path = epoch_path_prefix + "_dict.pth"
        save_path = epoch_path_prefix + ".pth"
        
        torch.save(model.state_dict(), dict_save_path)
        # torch.save(model, save_path)
    
    if avg_vloss<best_vloss:
        best_vloss = avg_vloss
        dict_save_path = save_path_prefix + "_best_dict.pth"
        save_path = save_path_prefix + ".pth"
        
        torch.save(model.state_dict(), dict_save_path)
        # torch.save(model, save_path)
        patience_count = 0
    else:
        patience_count += 1
        
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    epoch_number += 1

    if patience_count >= 10:
        break


print("Best validation loss -", best_vloss)