import os
import shutil
import re
import random

import torch
import torch.nn as nn
from torch import optim
from time import time
import torch.utils.data
from torch.utils.data import ConcatDataset
import torch.utils.data as data_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import itertools

from src.utils import PaintingsFolder
from src.began_pytorch_good import clean_dataset

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, channels_img, features_gen):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(channels_img, features_gen, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2 , inplace=True),
            Generator._default_block_downsample(features_gen, features_gen * 2, 4, 2, 1),
            Generator._default_block_downsample(features_gen * 2, features_gen * 4, 4, 2, 1),
            Generator._default_block_downsample(features_gen * 4, features_gen * 4, 4, 2, 1),
        )

        self.upsample = nn.Sequential(
            Generator._default_block_upsample(features_gen * 4, features_gen * 4, 4, 1, 1),
            Generator._default_block_upsample(features_gen * 4, features_gen * 2, 4, 2, 1),
            Generator._default_block_upsample(features_gen * 2, features_gen, 4, 2, 1),
            nn.ConvTranspose2d(features_gen, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    @staticmethod
    def _default_block_upsample(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    @staticmethod
    def _default_block_downsample(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2 , inplace=True)
        )

    def forward(self, x):
        intermediate = self.downsample(x)
        final = self.upsample(intermediate)

        return final

class Discriminator(nn.Module):
    def __init__(self, in_channels, features_d):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, features_d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            Discriminator._default_block(features_d, features_d * 2, 4, 2, 1),
            Discriminator._default_block(features_d * 2, features_d * 4, 4, 2, 1),
            Discriminator._default_block(features_d * 4, features_d * 4, 4, 2, 1),
            nn.Conv2d(features_d * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    @staticmethod
    def _default_block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.body(x)

class DISCOGAN:
    def __init__(self, dataloader_a, dataloader_b, batch_size=32, lr=0.002, device: torch.device ='cpu',
    reconstruction_criterion=nn.MSELoss(), gan_criterion=nn.BCELoss()): 

        self.generator_a2b = Generator(3, 64).to(device)
        self.generator_b2a = Generator(3, 64).to(device)
        self.generator_a2b.apply(weights_init)
        self.generator_b2a.apply(weights_init)

        self.discriminator_a = Discriminator(3, 64).to(device)
        self.discriminator_b = Discriminator(3, 64).to(device)
        self.discriminator_a.apply(weights_init)
        self.discriminator_b.apply(weights_init)
        
        self.optim_g = optim.Adam(itertools.chain(self.generator_a2b.parameters(), self.generator_b2a.parameters()), lr=lr, betas=(0.5, 0.999), weight_decay=0.00001)
        self.optim_d_a = optim.Adam(self.discriminator_a.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.00001)
        self.optim_d_b = optim.Adam(self.discriminator_b.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.00001)

        # how realistic a generated image is in domain B
        self.gan_criterion = gan_criterion

        # how well the original input is reconstructed after a sequence of 2 generations
        self.reconstruction_criterion = reconstruction_criterion

        self.dataloader_a = dataloader_a
        self.dataloader_b = dataloader_b
        self.batch_size = batch_size
        self.device = device

    def generate_images_a2b(self, images_vec=None):
        """
        Given a list of images from domain A, passes it to the generator 
        to obtain images associated to the domain B
        """
        with torch.no_grad():
            return self.generator_a2b(images_vec).detach().cpu()
    
    def generate_images_b2a(self, images_vec=None):
        """
        Given a list of images from domain B, passes it to the generator 
        to obtain images associated to the domain A
        """
        with torch.no_grad():
            return self.generator_b2a(images_vec).detach().cpu()

    def train_epoch(self):
        """Train both networks for one epoch and return the losses."""
        loss_g_running, loss_d_running = 0, 0
        for (images_a, images_b) in zip(self.dataloader_a, self.dataloader_b):
            
            data_a = images_a[0].to(self.device)
            data_b = images_b[0].to(self.device)

            ### GENERATOR TRAINING ###

            self.generator_a2b.train()
            self.generator_b2a.train()

            self.optim_g.zero_grad()

            AB = self.generator_a2b(data_a)
            BA = self.generator_b2a(data_b)

            ABA = self.generator_b2a(AB)
            BAB = self.generator_a2b(BA)

            pred_real_ba = self.discriminator_a(BA)
            pred_real_ab = self.discriminator_b(AB)
            GAN_loss = self.gan_criterion(pred_real_ba, torch.ones_like(pred_real_ba).to(self.device)) + self.gan_criterion(pred_real_ab, torch.ones_like(pred_real_ab).to(self.device))

            # Reconstruction Loss
            recon_loss_A = self.reconstruction_criterion(ABA, data_a)
            recon_loss_B = self.reconstruction_criterion(BAB, data_b)
            recon_loss = recon_loss_A + recon_loss_B

            loss_g = recon_loss + GAN_loss

            loss_g.backward()
            self.optim_g.step()

            ### DISCRIMINATOR TRAINING ###

            self.optim_d_a.zero_grad()
            self.optim_d_b.zero_grad()

            pred_a = self.discriminator_a(data_a)
            pred_b = self.discriminator_b(data_b)
            pred_a_fake = self.discriminator_a(BA.detach())
            pred_b_fake = self.discriminator_b(AB.detach())
            loss_d_a = self.gan_criterion(pred_a, torch.ones_like(pred_a).to(self.device)) * 0.5 + self.gan_criterion(pred_a_fake, torch.zeros_like(pred_a_fake).to(self.device)) * 0.5
            loss_d_b = self.gan_criterion(pred_b, torch.ones_like(pred_b).to(self.device)) * 0.5 + self.gan_criterion(pred_b_fake, torch.zeros_like(pred_b_fake).to(self.device)) * 0.5
            loss_d = loss_d_a + loss_d_b

            loss_d_a.backward()
            loss_d_b.backward()
            self.optim_d_a.step()
            self.optim_d_b.step()

            loss_d_running += loss_d
            loss_g_running += loss_g

        n_batches = min(len(self.dataloader_a), len(self.dataloader_b))
        loss_g_running /= n_batches
        loss_d_running /= n_batches

        return loss_g_running, loss_d_running

if __name__ == '__main__':
    shutil.rmtree("discogan_test_pytorch_art", ignore_errors=True)
    os.makedirs("discogan_test_pytorch_art")

    resized_images_dir = '../dataset/best_artworks/resized/resized'
    image_size = 64
    batch_size = 32

    epochs = 200

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    metadata_csv = pd.read_csv('../dataset/best_artworks/artists.csv')

    death_monet = 1926
    impressionist_artists_dict = dict()
    other_artists_to_consider_dict = dict()
    for artist_id, artist_name, artist_years, artistic_movement in zip(metadata_csv['id'],
                                                                       metadata_csv['name'],
                                                                       metadata_csv['years'],
                                                                       metadata_csv['genre']):

        dob = artist_years.split(' ')[0]
        if re.search(r'impressionism', artistic_movement.lower()):

            impressionist_artists_dict[artist_name] = artist_id
        elif int(dob) < death_monet:
            other_artists_to_consider_dict[artist_name] = artist_id

    clean_dataset(resized_images_dir)

    train_impressionist = PaintingsFolder(
        root=resized_images_dir,
        transform=transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),

            # normalizes images in range [-1,1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

        ]),
        artists_dict=impressionist_artists_dict
    )

    train_impressionist_augment1 = PaintingsFolder(
        root=resized_images_dir,
        transform=transforms.Compose([
            transforms.TrivialAugmentWide(),
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),

            # normalizes images in range [-1,1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

        ]),
        artists_dict=impressionist_artists_dict
    )

    train_others = PaintingsFolder(
        root=resized_images_dir,
        transform=transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),

            # normalizes images in range [-1,1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

        ]),
        artists_dict=other_artists_to_consider_dict
    )

    # Ugly for loop but better for efficiency, we save 5 random paintings for each 'other artist'.
    # we exploit the fact that first we have all paintings of artist x, then we have all paintings of artist y, etc.
    subset_idx_list = []
    current_artist_id = train_others.targets[0]
    idx_paintings_current = []
    for i in range(len(train_others.targets)):
        if train_others.targets[i] != current_artist_id:
            idx_paintings_to_hold = random.sample(idx_paintings_current, 5)
            subset_idx_list.extend(idx_paintings_to_hold)

            current_artist_id = train_others.targets[i]
            idx_paintings_current.clear()
        else:
            indices_paintings = idx_paintings_current.append(i)

    train_others = torch.utils.data.Subset(train_others, subset_idx_list)

    # concat original impressionist, augmented impressionist, 5 random paintings of other artists
    dataset_a = ConcatDataset([train_impressionist, train_impressionist_augment1, train_others])

    ds_path_b = "../dataset/photo_jpg"
    ds_path_test = "../dataset/photo_jpg_test"

    transformers=transforms.Compose([transforms.Resize(image_size), 
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset_b = dset.ImageFolder(root=ds_path_b, transform=transformers)

    if len(dataset_a) < len(dataset_b):
        indices = torch.arange(len(dataset_a))
        dataset_b = data_utils.Subset(dataset_b, indices)
    elif len(dataset_a) > len(dataset_b):
        indices = torch.arange(len(dataset_b))
        dataset_a = data_utils.Subset(dataset_a, indices)

    # Create the dataloader
    dataloader_a = torch.utils.data.DataLoader(dataset_a, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_b = torch.utils.data.DataLoader(dataset_b, batch_size=batch_size, shuffle=True, num_workers=0)

    # function for the learning rate decay
    gan = DISCOGAN(dataloader_a, dataloader_b, batch_size=batch_size, device=device)

    start = time()

    test_set = dset.ImageFolder(root=ds_path_test, transform=transformers)

    test_set_list = []
    for i, (image, _) in enumerate(test_set):
        if i < 64:
            test_set_list.append(image.to(device))
        else:
            break
    test_set_list = torch.stack(test_set_list)

    for i in range(epochs):
        print(f"Epoch {i+1}; Elapsed time = {int(time() - start)}s")

        g_loss, d_loss = gan.train_epoch()

        print(f"G_loss -> {g_loss}, D_loss -> {d_loss}\n")

        images = gan.generate_images_b2a(test_set_list)
        
        ims = vutils.make_grid(images, padding=2, normalize=True)
        plt.axis("off")
        plt.title(f"Epoch {i+1}")
        plt.imshow(np.transpose(ims, (1, 2, 0)))
        plt.savefig(f'discogan_test_pytorch_art/epoch{i+1}.png')

            
            