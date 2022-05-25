import os
import shutil
import re
import random

import torch
import torch.nn as nn
from torch import optim
from time import time
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset

from src.utils import PaintingsFolder
from src.began_pytorch_good import clean_dataset

"""

Structure of the code mirrors the implementation done for dcgan

Code for both generator and critic is an adaptation of the code from this notebook:
https://www.kaggle.com/code/errorinever/wgan-gp

Ideally we should mirror the Generator and Discriminator done for our DCGAN but i had problems doing so
After removing the sigmoid layer from the Discriminator and adapting the rest of the code, the images generated from the wgan 
got progressively worse for each epoch when tested on cifar10 planes only with 100 epochs

"""


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    https://www.kaggle.com/code/errorinever/wgan-gp
    """
    def __init__(self, channels_noise, channels_img, features_gen):
        """
        :param channels_noise: ``int``, input latent space dimension
        :param channels_img: ``int``,  3 for RGB image or 1 for GrayScale
        :param features_gen: ``int``, num features of generator
        """
        super().__init__()
        self.body = nn.Sequential(
            Generator._default_block(channels_noise, features_gen * 4, 4, 1, 0),
            Generator._default_block(features_gen * 4, features_gen * 4, 4, 2, 1), 
            Generator._default_block(features_gen * 4, features_gen * 2, 4, 2, 1), 
            Generator._default_block(features_gen * 2, features_gen, 4, 2, 1), 
            nn.ConvTranspose2d(
                features_gen, channels_img, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    @staticmethod
    def _default_block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.body(x)


class Critic(nn.Module):
    """
    https://www.kaggle.com/code/errorinever/wgan-gp
    """
    def __init__(self, in_channels, features_d):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, features_d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            Critic._default_block(features_d, features_d * 2, 4, 2, 1),
            Critic._default_block(features_d * 2, features_d * 4, 4, 2, 1),
            Critic._default_block(features_d * 4, features_d * 4, 4, 2, 1),
            nn.Conv2d(features_d * 4, 1, kernel_size=4, stride=1, padding=0, bias=False)
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



class WGAN_GP:
    def __init__(self, latent_dim, noise_fn, dataloader,
                 batch_size=32, lr_d=0.002, lr_g=0.002, device: torch.device ='cpu',
                 n_repetitions=5, l=10):
        """
        WGAN_GP class

        n_repetitions controls the number of times the critic will be trained for each step (5-10 usually)

        l is the lambda parameter of the gradient penalty (penalty coefficient)
        (default value is 10 as proposed in the improved wgan training paper)
        """         

        self.generator = Generator(latent_dim, 3, 64).to(device)
        self.generator.apply(weights_init)

        self.critic = Critic(3, 64).to(device)
        self.critic.apply(weights_init)
        self.noise_fn = noise_fn
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.device = device
        self.optim_c = optim.Adam(self.critic.parameters(), lr=lr_d, betas=(0.5, 0.999))
        self.optim_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))

        self.n_repetitions = n_repetitions
        self.l = l
    
    def compute_gp(self, real_images, fake_images):
        """
        https://towardsdatascience.com/demystified-wasserstein-gan-with-gradient-penalty-ba5e9b905ead
        """
        batch_size = real_images.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=True)

        interpolation =  epsilon * real_images + (1 - epsilon) * fake_images

        interp_logits = self.critic(interpolation)
        grad_outputs = torch.ones_like(interp_logits)

        gradients = torch.autograd.grad(
            inputs=interpolation,
            outputs=interp_logits,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]

        gradients = gradients.view(len(gradients), -1)
        grad_norm = gradients.norm(2, dim=1)
        return ((grad_norm - 1) ** 2).mean()
    
    def wasserstein_gp_loss(self, real_images, fake_images, pred_real, pred_fake):
        gp = self.compute_gp(real_images, fake_images)
        # wasserstein loss formula combined with gradient penalty
        return pred_fake.mean() - pred_real.mean() + gp * self.l

    def generate_samples(self, latent_vec=None, num=None):
        """Sample images from the generator.
        Images are returned as a 4D tensor of values between -1 and 1.
        Dimensions are (number, channels, height, width). Returns the tensor
        on cpu.
        Args:
            latent_vec: A pytorch latent vector or None
            num: The number of samples to generate if latent_vec is None
        If latent_vec and num are None then use self.batch_size
        random latent vectors.
        """
        num = self.batch_size if num is None else num
        latent_vec = self.noise_fn(num) if latent_vec is None else latent_vec
        with torch.no_grad():
            samples = self.generator(latent_vec)
        samples = samples.detach().cpu()  # move images to cpu
        return samples

    def train_step_generator(self, current_batch_size):
        """Train the generator one step and return the loss."""
        self.generator.zero_grad()

        # create noise 
        latent_vec = self.noise_fn(current_batch_size)
        # create fake images based on that noise
        generated = self.generator(latent_vec)
        # classify the created images
        classifications = self.critic(generated).reshape(-1)
        loss = -1. * classifications.mean()
        loss.backward()
        self.optim_g.step()
        return loss.item()

    def train_step_critic(self, real_samples, current_batch_size):
        """
        Trains the critic n times for each training step (where n = number of repetitions)
        The n number can be modified by changing the n_repetitions attribute
        """
        mean_iter_loss = 0
        for _ in range(self.n_repetitions):
            self.critic.zero_grad()

            # classify (networks predicts if images are real or fake) real images using the critic
            pred_real = self.critic(real_samples).reshape(-1)

            # generated fake images
            latent_vec = self.noise_fn(current_batch_size)
            with torch.no_grad():
                fake_samples = self.generator(latent_vec)

            # predict the fake images using the critic
            pred_fake = self.critic(fake_samples.detach()).reshape(-1)

            loss = self.wasserstein_gp_loss(real_samples, fake_samples, pred_real, pred_fake)
            mean_iter_loss += loss.item()

            loss.backward(retain_graph=True)
            self.optim_c.step()
        return mean_iter_loss / self.n_repetitions

    def train_epoch(self):
        """Train both networks for one epoch and return the losses."""
        loss_g_running, loss_c_running = 0, 0
        for _, (real_samples, _) in enumerate(self.dataloader):

            current_batch_size = real_samples.size(0)

            real_samples = real_samples.to(self.device)
            loss_c_running += self.train_step_critic(real_samples, current_batch_size)

            loss_g_running += self.train_step_generator(current_batch_size)

        n_batches = len(self.dataloader)
        loss_g_running /= n_batches
        loss_c_running /= n_batches

        return loss_g_running, loss_c_running


if __name__ == '__main__':
    shutil.rmtree("wgan_test_pytorch", ignore_errors=True)
    os.makedirs("wgan_test_pytorch")

    resized_images_dir = '../dataset/best_artworks/resized/resized'
    image_size = 64
    batch_size = 64
    epochs = 100 # ideally 200k as stated in the wgan-gp paper
    latent_dim = 100

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
    dataset = ConcatDataset([train_impressionist, train_impressionist_augment1, train_others])

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    # noise_fn is the function which will sample the latent vector from the gaussian distribution in this case
    noise_fn = lambda x: torch.normal(mean=0, std=1, size=(x, latent_dim, 1, 1), device=device)
    gan = WGAN_GP(latent_dim, noise_fn, dataloader, batch_size=batch_size, device=device)

    start = time()
    static_noise = torch.randn(64, 100, 1, 1, device=device)
    for i in range(epochs):
        print(f"Epoch {i+1}; Elapsed time = {int(time() - start)}s")

        g_loss, c_loss = gan.train_epoch()

        print(f"G_loss -> {g_loss}, C_loss -> {c_loss}\n")

        with torch.no_grad():
            images = gan.generator(static_noise)
        images = images.detach().cpu()
        ims = vutils.make_grid(images, padding=2, normalize=True)
        plt.axis("off")
        plt.title(f"Epoch {i+1}")
        plt.imshow(np.transpose(ims, (1, 2, 0)))
        plt.savefig(f'wgan_test_pytorch/epoch{i+1}.png')

