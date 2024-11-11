import os
import numpy as np
import torch
import time

from tqdm import tqdm

from mesmer import mesmer_preprocess
from iter_semantic import SemanticDataGenerator
from iter_cropping import CroppingDataGenerator

torch.cuda.empty_cache()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
):
    # use augmentation for training but not validation
    datagen = CroppingDataGenerator(
        rotation_range=rotation_range,
        shear_range=shear_range,
        zoom_range=(zoom_min, 1 / zoom_min),
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        crop_size=(crop_size, crop_size),
    )

    datagen_val = SemanticDataGenerator(
        rotation_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=0,
        vertical_flip=0,
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

seed = 0
zoom_min = 0.75
batch_size = 8

(X_train, y_train), (X_val, y_val) = load_data(tissuenet_dir)
X_test, y_test = _load_npz(os.path.join(tissuenet_dir, "test_256x256.npz"))

smaller = 30
smaller_test = None
if smaller:
    X_train, y_train = X_train[:smaller], y_train[:smaller]
    X_val, y_val = X_val[:smaller], y_val[:smaller]
if smaller_test:
    X_test, y_test = X_test[:smaller_test], y_test[:smaller_test]

train_dict = {"X": mesmer_preprocess(X_train), "y": y_train}
val_dict = {"X": mesmer_preprocess(X_val), "y": y_val}

print(train_dict["X"].shape)

train_data, val_data = create_data_generators(
    train_dict,
    val_dict,
    seed=seed,
    zoom_min=zoom_min,
    batch_size=batch_size,
    crop_size=crop_size,
)

def train_one_epoch(model):
    running_loss_avg = 0.
    count = 0

    per_epoch_steps = len(X_train)//batch_size if len(X_train)%batch_size==0 else (len(X_train)//batch_size + 1)
    for _ in tqdm(range(per_epoch_steps)):
        count += 1
        
        li_inputs, li_labels = train_data.next()
        
        inputs = np.transpose(li_inputs, (0, 3, 1, 2))
        labels = [np.transpose(l, (0, 3, 1, 2)) for l in li_labels]

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

model = model.to(device)

save_path_prefix = "/data/saved_model"
if smaller is None:
    save_path_prefix = save_path_prefix + "_full_gen_" + str(batch_size)
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
        for _ in tqdm(range(per_epoch_steps)):
            count += 1
            
            li_inputs, li_labels = val_data.next()

            vinputs = np.transpose(li_inputs, (0, 3, 1, 2))
            vlabels = [np.transpose(l, (0, 3, 1, 2)) for l in li_labels]
            
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
        dict_save_path = save_path_prefix + "_dict.pth"
        save_path = save_path_prefix + ".pth"
        
        torch.save(model.state_dict(), dict_save_path)
        # torch.save(model, save_path)
        
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    epoch_number += 1


print("Best validation loss -", best_vloss)