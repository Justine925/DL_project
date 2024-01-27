import torch
import numpy as np

def train_model(model, criterion, optimizer, trainloader, device, nb_epoch=2):
    """Trains a PyTorch model using the specified criterion and optimizer."""
    # Set the model to training mode
    model.train()
    # Loop over the specified number of epochs
    for epoch in range(nb_epoch):  
        print("Epoch", epoch)
        running_loss = 0.0
        # Loop over the mini-batches in the training loader
        for i, data in enumerate(trainloader, 0):
            # Get inputs and labels from the mini-batch
            inputs, labels = data
            # Move data to the specified device
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the gradients
            optimizer.zero_grad()
            # Pass inputs through the model
            outputs = model(inputs)
            # Compute the loss between model outputs and labels 
            loss = criterion(outputs, labels)
            # Compute gradients for each model parameter
            loss.backward()
            # Update model weights using the optimizer
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                print(f'[{epoch}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print('Finished Training')

def test_model(model, testloader, device):
    """Evaluates the performance of a PyTorch model on a test dataset."""
    # Set the model to evaluation mode
    model.eval()
    # Initialise counters
    conf_matrix = np.zeros((37, 37))
    correct = 0
    total = 0
    # Disable gradient computation during testing
    with torch.no_grad():
        # Loop over the data in the test loader
        for data in testloader:
            # Get inputs and labels from the mini-batch
            inputs, labels = data
            # Move data to the specified device
            inputs, labels = inputs.to(device), labels.to(device)
            #Pass inputs through the model
            outputs = model(inputs)
            # Choose the class with the highest energy as prediction
            _, predicted = torch.max(outputs.data, 1)
            # Update counters
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Update confusion matrix
            for i in range(len(predicted)):
                conf_matrix[int(predicted[i])][int(labels[i])] +=1
    return total, correct, conf_matrix