import torch
import torchvision.transforms as transforms
from data_handling import ImageDataset, getAnnotLists, getSetsIndices
from torch.utils.data import Subset
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

### TRANSFOMATIONS ###
# Define two data transformations
data_transform1 = transforms.Compose(
    [transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
data_transform2 = transforms.Compose(
    [ transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.RandomVerticalFlip(p=0.1),  
    transforms.RandomRotation(degrees=(-10, 10)),    
    transforms.ToTensor(),])

transform_map = {
    1 : data_transform1,
    2 : data_transform2
}

def get_train_dataloader(transform_id = 1, batch_size = 4):
    """Get the training DataLoader with the given transform and batch-size"""
    data_imgs = ImageDataset(root_dir='./Extracted_images/images/',
            transform=transform_map[transform_id])
    image_names,breeds_ids,species_ids = getAnnotLists()
    #Get the lists of indices for the sets
    train_indices, test_indices = getSetsIndices(data_imgs,breeds_ids)
    #Get the subset for images and annotations
    train_imgs = Subset(data_imgs, train_indices)
    train_breedsId = Subset(breeds_ids,train_indices)
    train_imgs = torch.stack([torch.tensor(img, dtype=torch.float32) for img in train_imgs])
    train_breeds = torch.tensor(train_breedsId, dtype=torch.long)
    #Create the dataset and the loader for training
    train_dataset = TensorDataset(train_imgs, train_breeds)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return trainloader

def get_test_dataloader(transform_id, batch_size):
    """Get the testing DataLoader"""
    data_imgs = ImageDataset(root_dir='./Extracted_images/images/',
            transform=transform_map[transform_id])
    image_names,breeds_ids,species_ids = getAnnotLists()
    #Get the lists of indices for the sets
    train_indices, test_indices = getSetsIndices(data_imgs,breeds_ids)
    #Get the subset for images and annotations
    test_imgs = Subset(data_imgs, test_indices)
    test_breedsId = Subset(breeds_ids,test_indices)
    test_imgs = torch.stack([torch.tensor(img, dtype=torch.float32) for img in test_imgs])
    test_breeds = torch.tensor(test_breedsId, dtype=torch.long)
    #Create the dataset and the loader for testing
    test_dataset = TensorDataset(test_imgs, test_breeds)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return testloader

### CRITERION ###
#Choice of the loss criterion used during training
def get_criterion(i=1):
    switch_dict = {
        1: nn.CrossEntropyLoss(),
        2: nn.NLLLoss(), #Negative Log Likelihood Loss - NLLLoss
    }
    return switch_dict.get(i)

### OPTIMIZER ###
#Choice of the optimization algorithm that updates the model's parameters during training.
def get_optimizer(model_param, i=1, lr=0.001, momentum=0.9):
    switch_dict = {
        1: optim.SGD(model_param, lr, momentum), #Stochastic Gradient Descent
        2: torch.optim.Adam(model_param, lr), #Adaptive Moment Estimation
        3: torch.optim.Adagrad(model_param, lr),
        4: torch.optim.Adadelta(model_param, lr),
        5: torch.optim.RMSprop(model_param, lr) #Root Mean Square Propagation
    }
    return switch_dict.get(i)
