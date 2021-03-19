from data_utils import prepare_datasets
from models.unet import UNet
from torch.utils.data import DataLoader

import torch
from torch import nn
import numpy as np

def train():
    # Download Data

    # Init data
    train_dataset, val_dataset = prepare_datasets()
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10)
    loaders = dict(train=train_loader, val=val_loader)

    # Init Model
    model = UNet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.984)
    loss_fn = nn.BCELoss()

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

            if phase == 'train':
                scheduler.step()

            # Функция, которая бы рисовала бы схему: картинка / правильный ответ / предикт и заливала бы её на гугл драйв
train()