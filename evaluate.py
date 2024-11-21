import os
import torch

from mesmer import Mesmer

from file_utils import _load_npz
from evaluate_utils import evaluate
from model_utils import create_model

# Set torch device
torch.cuda.empty_cache()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Set data directory
tissuenet_dir = "/data/tissuenet"
if not os.path.exists(tissuenet_dir):
    print("Created tissuenet data dir")
    os.makedirs(tissuenet_dir)
print(os.listdir(tissuenet_dir))

# Get evaluation data
X_test, y_test = _load_npz(os.path.join(tissuenet_dir, "test_256x256.npz"))
smaller_test = None
if smaller_test:
    X_test, y_test = X_test[:smaller_test], y_test[:smaller_test]

# Instantiate model
crop_size = 256
backbone = 'resnet50'
lr = 0.0001
model, losses, optimizer = create_model(
    input_shape=(crop_size, crop_size, 2),
    backbone=backbone,
    lr=lr,
    device=device,
)

# Load saved model
dict_save_path = "/data/saved_model_full_torch_8_best_dict.pth"
model.load_state_dict(torch.load(dict_save_path, map_location=device, weights_only=True))
model.to(device)
app = Mesmer(model)

# Evaluate
cell_preds = app.predict(X_test, batch_size=16)
cell_metrics = evaluate(cell_preds, y_test[..., :1])

nuc_preds = app.predict(X_test, batch_size=16, compartment="nuclear")
nuc_metrics = evaluate(nuc_preds, y_test[..., 1:])
