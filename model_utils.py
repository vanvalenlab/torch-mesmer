from panoptic import PanopticNet
import torch
import numpy as np
from loss_utils import semantic_loss

def create_model(input_shape=(256, 256, 2), backbone="resnet50", lr=1e-4, device=torch.device("cpu")):
    
    num_semantic_classes = [1, 3, 1, 3]  # inner distance, pixelwise, inner distance, pixelwise
    model = PanopticNet(
        backbone=backbone,
        input_shape=input_shape,
        norm_method=None,
        num_semantic_heads=4,
        num_semantic_classes=num_semantic_classes,
        location=True,  # should always be true
        include_top=True,
    )
    print("Model is using", device)

    loss = []
    model = model.to(device)
    for n_classes in num_semantic_classes:
        loss.append(semantic_loss(n_classes))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, loss, optimizer