import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict
from torch.utils.data import DataLoader

from workspace_utils import active_session


# Editting the class itself to add hidden layers array
# class Network(nn.Module):
#     def __init__(self, input_size, output_size, hidden_layers, dropout_p):
#         super().__init__()
        
#         # Input to a hidden layer
#         self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
#         # Adding more hidden layers
#         layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
#         self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
#         self.output = nn.Linear(hidden_layers[-1], output_size)

#         self.dropout = nn.Dropout(p= dropout_p)
        
#     def forward(self, x):        
#         for each in self.hidden_layers:
#             x = F.relu(each(x))
#             x = self.dropout(x)
#         x = self.output(x)
        
#         return F.log_softmax(x, dim = 1)

def load_data(data_dir, gpu):
    device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    # Define the paths for the data directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define the transformations for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    # Define the dataloaders for the training, validation, and testing sets
    train_loader = DataLoader(train_datasets, batch_size=64, shuffle=True)
    val_loader = DataLoader(valid_datasets, batch_size=32)
    test_loader = DataLoader(test_datasets, batch_size=32)

    img={"train":train_datasets,"test":test_datasets,"valid":valid_datasets}
    
    return img,train_loader, val_loader, test_loader


# Build model
def build_model(arch, hidden_size):
    # Load in a pre-trained model, default is DenseNet
    if arch.lower() == "vgg13":
        model = models.vgg13(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False 

    input_size = 25088 if arch.lower() == "vgg13" else 1024

    classifier = nn.Sequential(OrderedDict([
                        ('dropout1', nn.Dropout(0.1)),
                        ('fc1', nn.Linear(input_size, hidden_size)),
                        ('relu1', nn.ReLU()),
                        ('dropout2', nn.Dropout(0.1)),
                        ('fc2', nn.Linear(hidden_size, 102)),
                        ('output', nn.LogSoftmax(dim=1))
                        ]))

    model.classifier = classifier
    return model

# Validation to be done on each
def validation(model, dataloader, criterion, device):
    loss, accuracy = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss += criterion(output, labels).item()
            equality = (labels == output.max(dim=1)[1])
            accuracy += equality.float().mean()
    
    return loss, accuracy

            
#Doing the trainig and performace calulations in one to accomodate more model types

def train_model(model, train_loader, valid_loader, learning_rate, epochs, gpu):

    # Criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Train the classifier layers using backpropagation using the pre-trained network to get the features
    device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    print(type(model))
    model.to(device)
    print_every = 40
    steps = 0
    running_loss = 0
    train_accuracy = 0

    print("Training in progress using",device)

    with active_session():
        for e in range(epochs):

            model.train() # Dropout is turned on for training

            for images, labels in iter(train_loader):

                images, labels = images.to(device), labels.to(device) # Move input and label tensors to the GPU

                steps += 1
                optimizer.zero_grad()
                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                # get the class probabilities from log-softmax
                ps = torch.exp(output) 
                equality = (labels.data == ps.max(dim=1)[1])
                train_accuracy += equality.type(torch.FloatTensor).mean()

                if steps % print_every == 0:

                    model.eval() # Make sure network is in eval mode for inference

                    with torch.no_grad():
                        valid_loss, valid_accuracy = validation(model, valid_loader, criterion, device)

                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                        "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                        "Training Accuracy: {:.3f}".format(train_accuracy/print_every),
                        "Validation Loss: {:.3f}.. ".format(valid_loss/len(valid_loader)),
                        "Validation Accuracy: {:.3f}".format(valid_accuracy/len(valid_loader)))

                    running_loss = 0
                    train_accuracy = 0
                    model.train() # Make sure training is back on
                    
        print("\nTraining completed!")
    
    return model, optimizer, criterion
