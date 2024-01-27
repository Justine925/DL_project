import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn as nn

def getAlexNetModel(device, freeze = False):
    # Load a pre-trained AlexNet model
    alexnet = models.alexnet(pretrained=True)
    if freeze == True:
        # Freeze the parameters of the pre-trained model to prevent them from being updated during training
        for param in alexnet.parameters():
            param.requires_grad = False
    # Change last layer of the model
    num_ftrs = alexnet.classifier[6].in_features # get the input dimension of last layer
    alexnet.classifier[6] = nn.Linear(num_ftrs,37)
    alexnet = alexnet.to(device)
    return alexnet

def getResNetModel(device, freeze = False):
    # Load a pre-trained AlexNet model
    resnet = models.resnet18(pretrained=True)
    if freeze == True:
        # Freeze the parameters of the pre-trained model to prevent them from being updated during training
        for param in resnet.parameters():
            param.requires_grad = False
    # Change last layer of the model
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 37)
    resnet = resnet.to(device)
    return resnet

def getVGGModel(device, freeze = False):
    # Load a pre-trained VGG model
    vgg = models.vgg16(pretrained=True)
    if freeze == True:
        # Freeze the parameters of the pre-trained model to prevent them from being updated during training
        for param in vgg.parameters():
            param.requires_grad = False
    # change last layer of the model
    num_ftrs = vgg.classifier[6].in_features # get the input dimension of last layer
    vgg.classifier[6] = nn.Linear(num_ftrs,37)
    vgg = vgg.to(device)
    return vgg

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # fully connected blocks
        self.fc1 = nn.Sequential(nn.Linear(100352, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(512, 256),
                                 nn.BatchNorm1d(256),
                                 nn.ReLU(),
                                 nn.Dropout(0.5))
        self.fc3 = nn.Sequential(nn.Linear(256, 128),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.Dropout(0.5))
        self.fc4 = nn.Linear(128, 37)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

def getCustomModel(device):
    net = Net().to(device)
    net.apply(init_weights)
    return net