import argparse
import json

import torch

from beforePred import process_image, predict, imshow

# flowers/train/54/image_05403.jpg
# Get the command line input into the script
parser = argparse.ArgumentParser()
parser.add_argument('image_path', help='Path to image file')
parser.add_argument('checkpoint', default = '.', help='Path to checkpoint file')
parser.add_argument('--top_k', type=int, default=5, help='Number of top classes to return')
parser.add_argument('--category_names', default='cat_to_name.json', help='Path to JSON file containing flower names')
parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
args = parser.parse_args()

#Setting the device
device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

# Label mapping
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f, strict=False)
print("Data Loaded")

# Load the checkpoint
filepath = args.checkpoint + '/checkpoint.pth'
checkpoint = torch.load(filepath, map_location='cpu')
model = checkpoint["model"]
model.load_state_dict(checkpoint['state_dict'])
print("Checkpoint Loaded")

# Image preprocessing
np_image = process_image(args.image_path)
print("Image processed")
# imshow(np_image)

# Predict class and probabilities
probs, classes = predict(np_image, model, args.top_k, device)
classes_name = [cat_to_name[class_i] for class_i in classes]

#Printing the predictions
print(f"The top {args.top_k} flower predictions for '{args.image_path}' ")
print('---')
for i in range(len(probs)):
    print(f"{classes_name[i]} ({round(probs[i], 3)})")
print("---")