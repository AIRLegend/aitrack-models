from torchvision.models import resnet18, ResNet18_Weights

from model import Resnet18
from utils import show_grid_samples
from metrics import nme
from dataset import BWLS3D
from transforms import RandomRotation



from tqdm import trange
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import torch.optim as optim
import torch
import pandas as pd
import numpy as np
from PIL import Image


EPOCHS = 100
BATCH_SIZE = 128
DATALOADER_N_WORKERS = 4


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data = pd.read_csv('datasets/LS3D-W/splits.csv')


backbone = resnet18(num_classes=512)
backbone.conv1 = torch.nn.Conv2d(
    1, 64, kernel_size=7, stride=2, padding=3, bias=False
)  # one channel only.


model = Resnet18(backbone).to(device)
trainable_params = model.parameters()

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(trainable_params, lr=1e-3)


dataset_train = BWLS3D(
    data[data.is_train == 1],
    transforms=[
        RandomRotation(rotation_prob=.5),
    ]
)

dataset_val= BWLS3D(
    data[data.is_train == 0],
    transforms=[
        RandomRotation(rotation_prob=.3),
    ]
)



image_loader = DataLoader(dataset_train, 
                          batch_size  = BATCH_SIZE, 
                          shuffle     = True, 
                          num_workers = DATALOADER_N_WORKERS,
                          pin_memory  = True)

val_image_loader = DataLoader(dataset_val, 
                          batch_size  = BATCH_SIZE, 
                          shuffle     = True, 
                          num_workers = DATALOADER_N_WORKERS,
                          pin_memory  = True)



writer = SummaryWriter(comment='BW_Regre')

pbar = trange(len(image_loader))


num_train_batches = np.ceil(len(dataset_train) / BATCH_SIZE)
num_val_batches = np.ceil(len(dataset_val) / BATCH_SIZE)


train_losses = []

for epoch in range(EPOCHS):

    running_loss = 0.0
    running_nme = 0.0

    model.train(mode=True)
    
    for i, batch in enumerate(image_loader):
        imgs = batch['image'].to(torch.float).to(device)
        lms = batch['landmarks'].to(device)
        ds = batch['d'].to(device)
            
        # Write inputs to tensorboard
        # if i == 0:
        #     denormed = imgs.clone()
        #     for i in range(256):
        #         denormed[i] = denormalizer(denormalizer(denormed[i]))
        #     writer.add_images("input_imgs", denormed, epoch)
        #     if denormed.min() < 0 or denormed.max() > 1:
        #         import pdb; pdb.set_trace()
        #     break

        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs.float(), lms.float())

        loss.backward()
        optimizer.step()

        running_loss += loss.item() / (num_train_batches)
        running_nme += nme(outputs, lms, ds).item() / (num_train_batches)

        pbar.update()
        
        pbar.set_description(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {running_loss}", refresh=True)
        

    model.train(mode=False)
    
    # Run validation
    val_running_loss = 0
    val_running_nme = 0

    for i, batch in enumerate(val_image_loader):
        imgs = batch['image'].to(torch.float).to(device)
        lms = batch['landmarks'].to(device)
        ds = batch['d'].to(device)

        with torch.no_grad():
            outputs = model(imgs)
            val_running_loss += criterion(outputs.float(), lms.float()).float() / num_val_batches
            val_running_nme += nme(outputs, lms, ds).float() / num_val_batches


    # reset prog bar
    pbar.refresh()  # force show last state
    pbar.reset() 

    
    fig_train = show_grid_samples(dataset_train, model=model)
    fig_val = show_grid_samples(dataset_val, model=model)

    # write to tensorboard
    writer.add_scalar("loss/train", running_loss, (epoch+1))
    writer.add_scalar("nme/train", running_nme, (epoch+1))

    writer.add_scalar("nme/val", val_running_nme, (epoch+1))
    writer.add_scalar("loss/val", val_running_loss, (epoch+1))

    writer.add_figure("preds/train", fig_train, epoch+1)
    writer.add_figure("preds/val", fig_val, epoch+1)

    # record losses
    train_losses.append(running_loss)

    # save checkpoints
    if epoch % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss,
            }, f"./checkpoints/model_checkpoint_{epoch}.pt")


pbar.close()
print('Finished Training!')