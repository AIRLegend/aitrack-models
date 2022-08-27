import os
import sys
import argparse

from typing import Dict, Union
from torch.cuda import check_error
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms

from model import Resnet18
from utils import show_grid_samples
from metrics import nme
from dataset import BWLS3D
import transforms as customtransforms



from tqdm import trange
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import torch.optim as optim
import torch
import pandas as pd
import numpy as np
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Trainer:
    def __init__(
        self, 
        model, 
        loss, 
        optimizer,
        train_dataloader,
        val_dataloader = None,
        extra_metrics: Union[Dict, None]= None,
        tensorboard_writer=None,
        save_checkpoint_every:int = 5,
        checkpoint_path = "checkpoints/",
        model_id = "model"
        
    ) -> None:
    
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.extra_metrics = extra_metrics


        self._train_samples_nb = len(self.train_dataloader.dataset)
        self._val_samples_nb = len(self.val_dataloader.dataset) if self.val_dataloader else -1


        self._train_batch_nb = int(np.ceil(self._train_samples_nb / self.train_dataloader.batch_size))
        if val_dataloader:
            self._val_batch_nb = int(np.ceil(self._val_samples_nb / self.val_dataloader.batch_size))

        self.tensorboard_writer = tensorboard_writer
        self.save_checkpoint_every = save_checkpoint_every
        self.checkpoint_path = checkpoint_path
        self.model_id=model_id
        
        self.last_epoch_ = 0  # used for recovering previous training sessions
        


    def train(self, epochs=10, early_stop=None):
        model = self.model
        pbar = trange(self._train_batch_nb)

        train_losses = []
        val_losses = []

        for epoch in range(self.last_epoch_, self.last_epoch_ + epochs):
            running_loss = 0.0
            running_metrics = {}

            model.train(mode=True)
            
            # Actual train
            for i, batch in enumerate(self.train_dataloader):
                loss, metrics = self._run_train_step(batch)

                running_loss += loss / self._train_batch_nb

                for m, v in metrics.items():
                    running_metrics[f'train/{m}'] = v / self._train_batch_nb

                running_metrics['train/loss'] = running_loss

                pbar.update()
                
                pbar.set_description(f"[Epoch {epoch+1}/{self.last_epoch_ + epochs}] Loss: {running_loss}", refresh=True)
            
            train_losses.append(running_loss)
            # Validation
            if not self.val_dataloader:
                continue

            running_val_loss = 0.0
            for i, batch in enumerate(self.val_dataloader):
                loss, metrics = self._run_val_step(batch)

                running_val_loss += loss / self._val_batch_nb

                for m, v in metrics.items():
                    running_metrics[f'val/{m}'] = v / self._val_batch_nb

                running_metrics['val/loss'] = running_val_loss

                pbar.set_description(f"[Epoch {epoch+1}/{self.last_epoch_ + epochs}] Val Loss: {running_val_loss}", refresh=True)

            val_losses.append(running_val_loss)

            # log metrics
            self._log_metrics(running_metrics, epoch)

            # log sample images
            self._log_sample_preds(epoch)

            # save checkpoints
            if epoch != 0 and epoch % self.save_checkpoint_every == 0:
                path = os.path.join(self.checkpoint_path, f'{self.model_id}_{epoch}.pt')
                self.save_checkpoint(path, epoch=epoch, loss=running_loss)

            # reset prog bar
            pbar.refresh()  # force show last state
            pbar.reset() 
    
        pbar.close()

        # save checkpoint for last epoch
        path = os.path.join(self.checkpoint_path, f'{self.model_id}_{epoch}.pt')
        self.save_checkpoint(path, epoch=epoch, loss=running_loss)
    

    def _run_val_step(self, batch):
        model = self.model

        imgs = batch['image'].to(torch.float).to(device)
        lms = batch['landmarks'].to(device)
        ds = batch['d'].to(device)

        self.model.train(False)
        with torch.no_grad():
            outputs = model(imgs)
            loss = self.loss(outputs.float(), lms.float())
        
        metrics = self._run_metrics(batch, outputs)
        self.model.train(True)

        return loss.item(), metrics

    def _run_train_step(self, batch):
        imgs = batch['image'].to(torch.float).to(device)
        lms = batch['landmarks'].to(device)
        ds = batch['d'].to(device)
        
        self.optimizer.zero_grad()

        outputs = self.model(imgs)
        loss = self.loss(outputs.float(), lms.float())

        loss.backward()
        self.optimizer.step()

        self.model.train(False)
        metrics = self._run_metrics(batch, outputs)
        self.model.train(True)

        return loss.item(), metrics

    def _run_metrics(self, batch, model_outputs):
        metrics = None
        if self.extra_metrics is not None:
            lms = batch['landmarks'].to(device)
            ds = batch['d'].to(device)

            with torch.no_grad():
                metrics = {k: v(model_outputs, lms, ds) for k, v in self.extra_metrics.items()}

        return metrics

    def _log_metrics(self, metrics, step):
        writer = self.tensorboard_writer

        for m, val in metrics.items():
            writer.add_scalar(m, val, step)

    def _log_sample_preds(self, step):
        writer = self.tensorboard_writer

        dataset_train = self.train_dataloader.dataset
        dataset_val = None if self.val_dataloader is None else self.val_dataloader.dataset

        fig_train = show_grid_samples(dataset_train, model=self.model)

        writer.add_figure("preds/train", fig_train, step)

        fig_val = None
        if dataset_val:
            fig_val = show_grid_samples(dataset_val, model=self.model)
            writer.add_figure("preds/val", fig_val, step)

    
    def save_checkpoint(self, path, epoch=0, loss=0):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path)

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.last_epoch_ = checkpoint['epoch']
        

        
def train(data_path, 
          epochs=10, bs=128, workers=4, lr=1e-3, 
          checkpoint_dir='./checkpoints/', save_every=5,
          from_checkpoint=None,
          ):

    data = pd.read_csv(data_path)


    backbone = resnet18(num_classes=512)
    backbone.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )  # one channel only.


    model = Resnet18(backbone).to(device)
    trainable_params = model.parameters()

    
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(trainable_params, lr=lr)


    dataset_train = BWLS3D(
        data[data.is_train == 1],
        transforms=[
            #customtransforms.RandomMirrorSample(p=.4),
            customtransforms.RandomShift(p=.3),
            customtransforms.RandomRotation(rotation_prob=.3),
            customtransforms.RandomContrastBrightness(p=.2),
            customtransforms.NormalizeSample(mean=0.445313569, std=0.2692461874),
        ]
    )

    dataset_val= BWLS3D(
        data[data.is_train == 0],
        transforms=[
            #customtransforms.RandomMirrorSample(p=.4),
            customtransforms.RandomRotation(rotation_prob=.3),
            customtransforms.NormalizeSample(mean=0.445313569, std=0.2692461874)
        ]
    )

    image_loader = DataLoader(dataset_train, 
                              batch_size  = bs, 
                              shuffle     = True, 
                              num_workers = workers,
                              pin_memory  = True)

    val_image_loader = DataLoader(dataset_val, 
                              batch_size  = bs, 
                              shuffle     = True, 
                              num_workers = workers,
                              pin_memory  = True)

    
    writer = SummaryWriter(comment='BW_Regre')

    trainer = Trainer(
        model, 
        criterion, 
        optimizer=optimizer, 
        train_dataloader=image_loader, 
        val_dataloader=val_image_loader,
        extra_metrics= {'nme': nme},
        tensorboard_writer=writer,
        checkpoint_path=checkpoint_dir,
        save_checkpoint_every=save_every,
    )

    if from_checkpoint:
        trainer.load_from_checkpoint(from_checkpoint)

    trainer.train(epochs=epochs, early_stop=None)

    print("Done!")


def main(args):
    train(
        data_path=args.custom_datapath,
        epochs=args.epochs,
        lr=args.lr,
        bs=args.bs,
        workers=args.data_workers,   
        from_checkpoint=args.from_checkpoint,
        save_every=args.save_every
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train script.")

    parser.add_argument('--custom_datapath',
                    default='datasets/LS3D-W/menpo.csv',
                    help='Path to the split file.')

    parser.add_argument('--epochs',
                    default=15,
                    type=int,
                    help='Number of epochs')

    parser.add_argument('--lr',
                    default=1e-3,
                    type=float,
                    help='Learning rate')

    parser.add_argument('--bs',
                    default=256,
                    type=int,
                    help='Batch size')
    
    parser.add_argument('--checkpoint_dir',
                    default='./checkpoints/',
                    help='Directory where checkpoint (.pt) will be saved')

    parser.add_argument('--save_every',
                    default=5,
                    type=int,
                    help='Epoch frequency for saving checkpoints')

    parser.add_argument('--data-workers',
                    default=4,
                    type=int,
                    help='How many workers for data loaders')

    parser.add_argument('--from-checkpoint',
                    default=None,
                    help='Resume training from previously saved checkpoint')

    args  = parser.parse_args(sys.argv[1:])
    
    main(args)