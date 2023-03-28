import numpy as np
import pandas as pd

import torch
from workspace_utils import active_session
from torch.utils.data import DataLoader
from collections import OrderedDict

from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from PIL import Image
import matplotlib.pyplot as plt


def process_image(image_path):
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    image = Image.open(image_path)
    image = image.resize((256, 256)).crop((16, 16, 240, 240))
    np_image = np.array(image) / 255
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    return np_image


def imshow(image, ax=None, title=None):
    ax = plt.subplots()[1] if ax is None else ax
    image = image.transpose((1, 2, 0))
    image = (image * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    ax.imshow(np.clip(image, 0, 1))
    return ax


def predict(image, model, topk=5, gpu='True'):
    device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')

    # Move the model to the device
    model.to(device)
    model.eval()

    # Convert the image to a tensor and move it to the device
    image_tensor = torch.from_numpy(image).to(device).float().unsqueeze(0)

    # Disable autograd to save memory and time
    with torch.no_grad():
        # Run the forward pass
        output = model(image_tensor)
        # Get the top-k predictions
        probs, indices = torch.topk(torch.exp(output), topk)
        # Convert the probabilities and indices to numpy arrays
        probs = probs.cpu().numpy().squeeze()
        indices = indices.cpu().numpy().squeeze()
        # Map the indices to class labels
        inv_map = {v: k for k, v in model.class_to_idx.items()}
        classes = [inv_map[int(index)] for index in indices]

    return probs.tolist(), classes
