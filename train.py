import argparse
import torch
from torchvision import datasets, transforms, models
from beforeTrain import load_data, build_model, train_model


parser = argparse.ArgumentParser(description='Train a neural network to classify flower images')
parser.add_argument('data_dir', type=str, help='Directory where the flower images are located')
parser.add_argument('--save_dir', type=str, default = '.', help='Directory to save model checkpoints')
parser.add_argument('--arch', type=str, default='densenet121', help='Name of pre-trained model architecture')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
parser.add_argument('--hidden_units', type=int, nargs='+', default=512, help='Number of nodes in hidden layers')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
parser.add_argument('--gpu',default='True', action='store_true', help='Use GPU for training')

args = parser.parse_args()


# Setting the device
device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

# Load data
image_datasets, train_loader, valid_loader, test_loader = load_data(args.data_dir, device)
print("Data loaded")

# Build model
model_init = build_model(args.arch, args.hidden_units)
print("Model built")

#Train model
model, optimizer, criterion = train_model(model_init, train_loader, valid_loader, args.learning_rate, args.epochs, device)
print("Model trained")

#Saving the checkpoint
path = args.save_dir +'/checkpoint.pth'
model.to('cpu')
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {'model': model,
              'state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict,
              'criterion': criterion,
              'epochs': args.epochs,
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, path)
print("Check point saved at",path)

'''
OLD STUFF --IGNORE--
# SAVING THE CHECKPOINT FOR VGG
# Create training_dict
input_size = model.classifier.hidden_layers[0].in_features
output_size = model.classifier.output.out_features
hidden_layers = []
for layer in model.classifier.hidden_layers:
        hidden_layers.append(layer.out_features)

dropout_p = model.classifier.dropout.p

for param_group in optimizer.param_groups:
    learning_rate = param_group['lr']

train_dict = {'input_size':input_size, 'output_size':output_size, 
                         'hidden_layers':hidden_layers,
                         'criterion':criterion, 'optimizer':optimizer, 
                         'learning_rate':learning_rate, 'dropout_p':dropout_p}

print(train_dict)

model.to('cpu')

checkpoint = {
    'model':model,
    'state_dict': model.state_dict(),
    'class_to_idx': img_datasets['train'].class_to_idx
}
'''