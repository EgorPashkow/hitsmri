import data_utils
from data_utils import prepare_datasets_FLAIR
from models.unet_flair import UNetFLAIR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import torch
from torch import nn
import numpy as np


def visualize_results(loader, model, epoch, phase):
    while True:
        batch = next(iter(loader))
        x, y = batch
        mask_sum = y.sum(dim=(1, 2, 3))
        where = torch.where(mask_sum)

        if len(where[0]):
            i = where[0][0]
            x = x[i:i + 1]
            y = y[i:i + 1]

            break

    model = model.eval()
    with torch.no_grad():
        pred = model(x.cuda())

    x = x.numpy()[0, ...].swapaxes(0, -1).swapaxes(0, 1).repeat(3, -1)
    pred = pred.detach().cpu().numpy()[0, 0, ...]
    y = y.detach().cpu().numpy()[0, 0, ...]
    intersect = pred.copy()
    intersect[y == 0] = 0

    pred = pred[..., None].repeat(3, -1)
    y = y[..., None].repeat(3, -1)
    intersect = intersect[..., None].repeat(3, -1)

    result = np.concatenate([x, y, pred, intersect], 1)
    plt.imsave(f'visualization_flair/{phase}_{epoch:03d}.png', result)

    return None

def tensorboard(losses, phase):
    plt.semilogy(losses)
    plt.savefig(f'{phase}_loss.png')

def train_flair(model_name=''):
    # Init data
    train_dataset, val_dataset = prepare_datasets_FLAIR()
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True)
    loaders = dict(train=train_loader, val=val_loader)

    # Init Model
    if model_name == '':
        model = UNetFLAIR().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.984)
        loss_fn = nn.BCELoss()
    else:
        model = data_utils.load_model(model_name)

    epochs = 500
    for epoch in range(epochs):
        for phase in 'train val'.split():
            if phase == 'train':
                model = model.train()
                torch.set_grad_enabled(True)

            else:
                model = model.eval()
                torch.set_grad_enabled(False)

            loader = loaders[phase]
            epoch_losses = dict(train=[], val=[])
            running_loss = []

            for batch in loader:
                imgs, masks = batch
                imgs = imgs.cuda()
                masks = masks.cuda()

                outputs = model(imgs)
                loss = loss_fn(outputs, masks)

                running_loss.append(loss.item())

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # End of Epoch
            print(f'{epoch}) {phase} loss: {np.mean(running_loss)}')
            visualize_results(loader, model, epoch, phase)

            if(epoch % 10 == 0):
                data_utils.save_model(model, f'weight_flair/model_{epoch}.pt')

            epoch_losses[phase].append(np.mean(running_loss))
            tensorboard(epoch_losses[phase], phase)

            if phase == 'train':
                scheduler.step()
