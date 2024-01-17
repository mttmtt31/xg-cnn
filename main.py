import torch.nn as nn
from src import train, val, train_val_split, load_model, set_optimiser, load_heatmap_dataset, save_model
from torch.utils.data import DataLoader
import wandb
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type = str, default = 'data/shots.npy', help = 'Complete path to numpy array of features')
    parser.add_argument('--labels', type = str, default = 'data/labels.npy', help = 'Complete path to numpy array of labels')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--batch-size', type = int, default = 64)
    parser.add_argument('--learning-rate', type = float, default = 0.0005, help = 'Set a static learning rate for the optimiser. In general, we noticed that small learning rates (<=0.001) have better performance.')
    parser.add_argument('--dropout', type = float, default = 0.0)
    parser.add_argument('--epochs', type = int, default = 50, help = 'Set the number epochs. The model usualy converges after 50 epochs.')
    parser.add_argument('--augmentation', action='store_true', help = 'Whether you want to perform data augmentation')
    parser.add_argument('--wandb', action='store_true', help = 'Whether you want to log results in wandb in a new project called xg-cnn. If specified, please make sure wandb is correctly configured on your end.')
    parser.add_argument('--save', action='store_true', help = 'Whether you want to save the model for later use. By default, it will be saved in the trained_models folder.')
    parser.add_argument('--optim', type = str, default = 'sgd', choices=['sgd', 'adam', 'adamw'], help = 'Optimiser to use')
    parser.add_argument('--weight-decay', type = float, default = 0.0)
    parser.add_argument('--gaussian-filter', type = float, default = 1.25, help = 'Variance of the Gaussian filter to apply on top of each freeze frame to smooth out its look. We noticed that values slightly larger than 1 have better performance.')

    return parser.parse_args()

def main(data, labels, device, batch_size, lr, num_epochs, g_filter, log_wandb, augmentation, dropout, optimiser, wd, save):
    # Load the dataset
    dataset = load_heatmap_dataset(data_path = data, labels_path = labels, augmentation=augmentation, g_filter=g_filter)
    # Split train/val dataset
    train_dataset, val_dataset = train_val_split(dataset=dataset, train_size=0.8)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = load_model(dropout=dropout)
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = set_optimiser(model=model, optim=optimiser, learning_rate=lr, weight_decay=wd)

    # log in wandb
    if log_wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="xg-cnn",
            
            # track hyperparameters and run metadata
            config={
                "learning_rate" : lr, 
                "epochs": num_epochs,
                "batch_size" : batch_size, 
                "dropout" : dropout,
                "augmentation" : augmentation,
                "optimiser" : optimiser,
                "weight_decay" : wd,
                "gaussian_filter" : g_filter
                }
        )
    # Train the model
    for i in range(num_epochs):
        model, train_loss = train(train_loader=train_loader, model=model, epoch=i, device=device, optimizer=optimizer, criterion=criterion)

        # evaluate the model on the validation set
        roc_score, log_loss, rmse = val(model=model, val_loader=val_loader, device=device, epoch=i, criterion=criterion)

        if log_wandb:
            wandb.log({"Train Loss": train_loss, "Validation ROC-AUC score:" : roc_score, "Validation loss" : log_loss, "Validation RMSE" : rmse})
        else:
            print(f'Epoch {i + 1}')
            print({"Train Loss": train_loss, "Validation ROC-AUC score:" : roc_score, "Validation loss" : log_loss, "Validation RMSE" : rmse})

    if save:
        save_model(model)

    if log_wandb:
        wandb.finish()

if __name__ == '__main__':
    args = parse_args()
    main(
        data=args.data,
        labels=args.labels,
        device=args.device,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        log_wandb=args.wandb,
        num_epochs=args.epochs,   
        g_filter=args.gaussian_filter,    
        augmentation=args.augmentation,
        dropout=args.dropout,
        optimiser=args.optim,
        wd = args.weight_decay,
        save = args.save
    )
