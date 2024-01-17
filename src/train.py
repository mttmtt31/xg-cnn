from tqdm import tqdm
import torch

def train(train_loader, model, epoch, device, optimizer, criterion):
    # sets into training mode
    model.train()
    # initialise parameter to track the training performance
    running_loss = 0.0

    # loop over the images
    for images, labels in tqdm(train_loader, total = len(train_loader), desc = f'Training epoch #{epoch+1}'):
        # send them to the device
        images, labels = images.to(device), labels.to(device)
        # forward pass
        outputs = model(images)
        # loss
        loss = criterion(outputs.squeeze(), labels.float())
        # zero the gradients
        optimizer.zero_grad()
        # backpropagation
        loss.backward()
        # optimisation
        optimizer.step()

        # track loss
        running_loss = running_loss + loss.item()

    # average the loss to get the training loss
    train_loss = running_loss / len(train_loader)

    return model, train_loss
