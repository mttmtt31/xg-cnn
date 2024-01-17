import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, mean_squared_error
import numpy as np

def val(model, val_loader, device, epoch, criterion):
    model.eval()
    xGs = []
    outcomes = []

    # initialise parameter to track the training performance
    log_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, total = len(val_loader), desc = f'Validating epoch #{epoch+1}'):
            # send to device
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs.squeeze(), labels.float())
            # track loss
            log_loss = log_loss + loss.item()

            # append the xG
            xGs.append(outputs.squeeze())
            outcomes.append(labels.float())

        # calculate the roc-auc score
        xGs = torch.cat(xGs).cpu()
        outcomes = torch.cat(outcomes).cpu()
        roc_auc = roc_auc_score(outcomes, xGs)

        # log-loss
        log_loss = log_loss / len(val_loader)

        # rmse
        rmse = np.sqrt(mean_squared_error(outcomes, xGs))
        
    return roc_auc, log_loss, rmse