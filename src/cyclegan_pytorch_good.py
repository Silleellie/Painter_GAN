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

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, channels_img, features_gen, num_res_blocks):
        super().__init__()

        layers = [nn.ReflectionPad2d(channels_img),
                  nn.Conv2d(channels_img, features_gen, 7, 1, 0),
                  nn.InstanceNorm2d(features_gen),
                  nn.ReLU(inplace=True)]
        
        layers.extend(self._default_block_downsample(features_gen, features_gen*2))
        layers.extend(self._default_block_downsample(features_gen*2, features_gen*4))
        layers.extend(self._default_block_downsample(features_gen*4, features_gen*4))

        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(features_gen*4))
        
        layers.extend(self._default_block_upsample(features_gen*4, features_gen*4))
        layers.extend(self._default_block_upsample(features_gen*4, features_gen*2))
        layers.extend(self._default_block_upsample(features_gen*2, features_gen))

        layers.extend([nn.ReflectionPad2d(channels_img),
                       nn.Conv2d(features_gen, channels_img, 7, 1, 0),
                       nn.Tanh()])
        
        self.model = nn.Sequential(*layers)

    @staticmethod
    def _default_block_upsample(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
        return [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True)
        ]

    @staticmethod
    def _default_block_downsample(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
        return [
                nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
        ]

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels, features_d):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, features_d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            Discriminator._default_block(features_d, features_d * 2, 4, 2, 1),
            Discriminator._default_block(features_d * 2, features_d * 4, 4, 2, 1),
            Discriminator._default_block(features_d * 4, features_d * 4, 4, 2, 1),
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(features_d * 4, 1, kernel_size=4, stride=1, padding=1, bias=False)
        )

    @staticmethod
    def _default_block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.body(x)

class CYCLEGAN:
    def __init__(self, dataloader_a, dataloader_b, lr_decay_func, batch_size=32, lr=0.002, device: torch.device ='cpu',
                 lambda_cycle=10.0, lambda_identity=5.0, num_res_blocks=6,
                 adversarial_criterion = nn.MSELoss(), cycle_criterion = nn.L1Loss(), identity_criterion = nn.L1Loss()):
        """
        suggested number of residual blocks equal to 6 for 128x128 images and equal to 9 for 256x256 or better quality images
        """

        self.generator_a2b = Generator(3, 64, num_res_blocks).to(device)
        self.generator_b2a = Generator(3, 64, num_res_blocks).to(device)
        self.generator_a2b.apply(weights_init)
        self.generator_b2a.apply(weights_init)

        self.discriminator_a = Discriminator(3, 64).to(device)
        self.discriminator_b = Discriminator(3, 64).to(device)
        self.discriminator_a.apply(weights_init)
        self.discriminator_b.apply(weights_init)
        
        self.optim_g = optim.Adam(itertools.chain(self.generator_a2b.parameters(), self.generator_b2a.parameters()), lr=lr, betas=(0.5, 0.999))
        self.optim_d_a = optim.Adam(self.discriminator_a.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optim_d_b = optim.Adam(self.discriminator_b.parameters(), lr=lr, betas=(0.5, 0.999))

        self.dataloader_a = dataloader_a
        self.dataloader_b = dataloader_b
        self.batch_size = batch_size
        self.device = device

        # parameters used in the computation of the total loss of the generator
        # in the CycleGan paper, in the task of converting Monet's paintings to photos, the value
        # for the cycle lambda was set to 10 while the suggested value for the lambda identity 
        # is equal to half the value of the cycle lambda (therefore in this case equal to 5)
        self.lambda_cycle = lambda_cycle # multiplied to the cycle loss
        self.lambda_identity = lambda_identity # multiplied to the identity loss

        # generator tries to minimize the loss while discriminator tries to maximize it
        self.adversarial_criterion = adversarial_criterion

        # encourages cycle consistency (the image translation cycle should be able to revert the image to its original state)
        self.cycle_criterion = cycle_criterion

        # encouranges the preservation of color composition
        self.identity_criterion = identity_criterion

        # used for the decay of the learning rate
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optim_g, lr_lambda=lr_decay_func)
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(self.optim_d_a, lr_lambda=lr_decay_func)
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optim_d_b, lr_lambda=lr_decay_func)
    
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
        """
        Train both networks for one epoch and return the losses.
        Reference used for epoch training code: https://www.kaggle.com/code/lmyybh/pytorch-cyclegan
        """
        loss_g_running, loss_d_running = 0, 0
        for (images_a, images_b) in zip(self.dataloader_a, self.dataloader_b):
            
            real_A = images_a[0].to(self.device)
            real_B = images_b[0].to(self.device)

            self.optim_g.zero_grad()

            ### GENERATOR TRAINING ###
        
            fake_B = self.generator_a2b(real_A)
            fake_A = self.generator_b2a(real_B)
            
            # identity loss
            loss_id_A = self.identity_criterion(fake_B, real_A)
            loss_id_B = self.identity_criterion(fake_A, real_B)
            loss_identity = loss_id_A + loss_id_B
            
            # adversarial loss
            pred_AB = self.discriminator_b(fake_B)
            pred_BA = self.discriminator_a(fake_A)
            loss_GAN_AB = self.adversarial_criterion(pred_AB, torch.ones_like(pred_AB)) 
            loss_GAN_BA = self.adversarial_criterion(pred_BA, torch.ones_like(pred_BA))
            loss_GAN = loss_GAN_AB + loss_GAN_BA
            
            # cycle loss
            recov_A = self.generator_b2a(fake_B)
            recov_B = self.generator_a2b(fake_A)
            loss_cycle_A = self.cycle_criterion(recov_A, real_A)
            loss_cycle_B = self.cycle_criterion(recov_B, real_B)
            loss_cycle = loss_cycle_A + loss_cycle_B

            loss_G = self.lambda_identity * loss_identity + loss_GAN + self.lambda_cycle * loss_cycle
            
            loss_G.backward()
            self.optim_g.step()
            
            # DISCRIMINATOR A TRAINING

            self.optim_d_a.zero_grad()
            
            pred_a = self.discriminator_a(real_A)
            pred_a_fake = self.discriminator_a(fake_A.detach())
            loss_real = self.adversarial_criterion(pred_a, torch.ones_like(pred_a))
            loss_fake = self.adversarial_criterion(pred_a_fake, torch.zeros_like(pred_a_fake))
            # divided by 2 as suggested in the CycleGan paper
            loss_D_A = (loss_real + loss_fake) / 2
            
            loss_D_A.backward()
            self.optim_d_a.step()
            
            # DISCRIMINATOR B TRAINING

            self.optim_d_b.zero_grad()
            
            pred_b = self.discriminator_b(real_B)
            pred_b_fake = self.discriminator_b(fake_B.detach())
            loss_real = self.adversarial_criterion(pred_b, torch.ones_like(pred_b))
            loss_fake = self.adversarial_criterion(pred_b_fake, torch.zeros_like(pred_b_fake))
            # divided by 2 as suggested in the CycleGan paper
            loss_D_B = (loss_real + loss_fake) / 2
            
            loss_D_B.backward()
            self.optim_d_b.step()

            loss_g_running += loss_G.item()
            loss_d_running += loss_D_A.item() + loss_D_B.item()
        
        # learning rate decay after epoch
        self.lr_scheduler_G.step()
        self.lr_scheduler_D_A.step()
        self.lr_scheduler_D_B.step()

        n_batches = min(len(self.dataloader_a), len(self.dataloader_b))
        loss_g_running /= n_batches
        loss_d_running /= n_batches

        return loss_g_running, loss_d_running

if __name__ == '__main__':
    """
    CYCLEGAN PAPER
    https://arxiv.org/abs/1703.10593
    """
    shutil.rmtree("cyclegan_test_pytorch_art_2", ignore_errors=True)
    os.makedirs("cyclegan_test_pytorch_art_2")

    resized_images_dir = '../dataset/best_artworks/resized/resized'
    image_size = 64
    batch_size = 32

    # as in the paper, after 100 epochs (out of 200) the learning rate should start decaying
    epochs = 200
    decay_epoch = 100

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
    lr_decay_func = lambda epoch: 1 - max(0, epoch-decay_epoch)/(epochs-decay_epoch)
    gan = CYCLEGAN(dataloader_a, dataloader_b, lr_decay_func, batch_size=batch_size, device=device)

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
        plt.savefig(f'cyclegan_test_pytorch_art_2/epoch{i+1}.png')
