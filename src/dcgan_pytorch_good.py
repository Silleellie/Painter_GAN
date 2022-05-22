import os
import random
import re
import shutil

import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from time import time
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset

from src.utils import PaintingsFolder


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        """A generator for mapping a latent space to a sample space.
        The sample space for this generator is single-channel, 32x32 images
        with pixel intensity ranging from -1 to +1.
        Args:
            latent_dim (int): latent dimension ("noise vector")
            batchnorm (bool): Whether or not to use batch normalization
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        # Project the input
        self.linear1 = nn.Linear(self.latent_dim, 512 * 2 * 2)
        self.bn1d1 = nn.BatchNorm1d(512 * 2 * 2)
        self.relu = nn.ReLU()

        # Convolutions
        self.conv1 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1)
        self.bn2d1 = nn.BatchNorm2d(256)

        # Convolutions
        self.conv2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1)
        self.bn2d2 = nn.BatchNorm2d(128)

        # Convolutions
        self.conv3 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1)
        self.bn2d3 = nn.BatchNorm2d(64)

        # Convolutions
        self.conv4 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1)
        self.bn2d4 = nn.BatchNorm2d(32)

        self.conv5 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1)
        self.tanh = nn.Tanh()

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        intermediate = self.linear1(input_tensor)
        intermediate = self.bn1d1(intermediate)
        intermediate = self.relu(intermediate)

        # reshape
        intermediate = intermediate.view((input_tensor.shape[0], 512, 2, 2))

        intermediate = self.conv1(intermediate)
        intermediate = self.bn2d1(intermediate)
        intermediate = self.relu(intermediate)

        intermediate = self.conv2(intermediate)
        intermediate = self.bn2d2(intermediate)
        intermediate = self.relu(intermediate)

        intermediate = self.conv3(intermediate)
        intermediate = self.bn2d3(intermediate)
        intermediate = self.relu(intermediate)

        intermediate = self.conv4(intermediate)
        intermediate = self.bn2d4(intermediate)
        intermediate = self.relu(intermediate)

        intermediate = self.conv5(intermediate)
        output_tensor = self.tanh(intermediate)
        return output_tensor


class Discriminator(nn.Module):
    def __init__(self):
        """A discriminator for discerning real from generated images.
        Images must be single-channel and 28x28 pixels.
        Output activation is Sigmoid.
        """
        super(Discriminator, self).__init__()
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout_2d = nn.Dropout2d(0.25)

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=True)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=True)
        self.bn2d2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=True)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128 * 8 * 8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """Forward pass; map samples to confidence they are real [0, 1]."""
        intermediate = self.conv1(input_tensor)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.dropout_2d(intermediate)

        intermediate = self.conv2(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.dropout_2d(intermediate)
        intermediate = self.bn2d2(intermediate)

        intermediate = self.conv3(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.dropout_2d(intermediate)

        # reshape
        intermediate = intermediate.view((input_tensor.shape[0], 128 * 8 * 8))
        intermediate = self.linear1(intermediate)
        output_tensor = self.sigmoid(intermediate)

        return output_tensor


class DCGAN:
    def __init__(self, latent_dim, noise_fn, dataloader,
                 batch_size=32, lr_d=0.0002, lr_g=0.0002, device: torch.device = 'cpu'):
        """A very basic DCGAN class for generating MNIST digits
        Args:
            generator: a Ganerator network
            discriminator: A Discriminator network
            noise_fn: function f(num: int) -> pytorch tensor, (latent vectors)
            dataloader: a pytorch dataloader for loading images
            batch_size: training batch size. Must match that of dataloader
            device: cpu or CUDA
            lr_d: learning rate for the discriminator
            lr_g: learning rate for the generator
        """
        self.generator = Generator(latent_dim).to(device)
        self.generator.apply(weights_init)

        self.discriminator = Discriminator().to(device)
        self.discriminator.apply(weights_init)

        self.noise_fn = noise_fn
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.device = device
        self.criterion = nn.BCELoss()
        self.optim_d = optim.Adam(self.discriminator.parameters(),
                                  lr=lr_d, betas=(0.5, 0.999))
        self.optim_g = optim.Adam(self.generator.parameters(),
                                  lr=lr_g, betas=(0.5, 0.999))
        self.real_labels = None
        self.fake_labels = None

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
        self.optim_g.zero_grad()

        latent_vec = self.noise_fn(current_batch_size)
        generated = self.generator(latent_vec)
        classifications = self.discriminator(generated)
        loss = self.criterion(classifications, self.real_labels)
        loss.backward()
        self.optim_g.step()
        return loss.item()

    def train_step_discriminator(self, real_samples, current_batch_size):
        """Train the discriminator one step and return the losses."""
        self.optim_d.zero_grad()

        pred_real = self.discriminator(real_samples)
        loss_real = self.criterion(pred_real, self.real_labels)

        # generated samples
        latent_vec = self.noise_fn(current_batch_size)
        with torch.no_grad():
            fake_samples = self.generator(latent_vec)
        pred_fake = self.discriminator(fake_samples)
        loss_fake = self.criterion(pred_fake, self.fake_labels)

        # combine two losses
        loss = (loss_real + loss_fake) / 2
        loss.backward()
        self.optim_d.step()
        return loss_real.item(), loss_fake.item()

    def train_epoch(self):
        """Train both networks for one epoch and return the losses.
        Args:
            print_frequency (int): print stats every `print_frequency` steps.
            max_steps (int): End epoch after `max_steps` steps, or set to 0
                             to do the full epoch.
        """
        loss_g_running, loss_d_real_running, loss_d_fake_running = 0, 0, 0
        for batch, (real_samples, _) in enumerate(self.dataloader):

            current_batch_size = real_samples.size(0)
            real_samples = real_samples.to(self.device)

            # We build labels here so that if the last batch has less samples
            # we don't have to drop it but we can still use it
            # we perform smooth labels
            self.real_labels = torch.ones((current_batch_size, 1), device=self.device)
            self.real_labels += 0.05 * torch.rand(self.real_labels.size(), device=self.device)

            self.fake_labels = torch.zeros((current_batch_size, 1), device=self.device)
            self.fake_labels += 0.05 * torch.rand(self.fake_labels.size(), device=self.device)

            loss_g_running += self.train_step_generator(current_batch_size)

            half_batch_size = int(current_batch_size / 2)

            random_idx = np.random.randint(0, current_batch_size, half_batch_size)
            random_half_batch = real_samples[random_idx]

            self.real_labels = torch.ones((half_batch_size, 1), device=self.device)
            self.real_labels += 0.05 * torch.rand(self.real_labels.size(), device=self.device)

            self.fake_labels = torch.zeros((half_batch_size, 1), device=self.device)
            self.fake_labels += 0.05 * torch.rand(self.fake_labels.size(), device=self.device)

            ldr_, ldf_ = self.train_step_discriminator(random_half_batch, half_batch_size)
            loss_d_real_running += ldr_
            loss_d_fake_running += ldf_

        n_batches = len(self.dataloader)
        loss_g_running /= n_batches
        loss_d_real_running /= n_batches
        loss_d_fake_running /= n_batches

        return loss_g_running, (loss_d_real_running, loss_d_fake_running)


if __name__ == '__main__':
    """
    GAN architecture inspired by
    
    https://towardsdatascience.com/how-to-build-a-dcgan-with-pytorch-31bfbf2ad96a
    
    result with 100 epoch:
    Epoch 100; Elapsed time = 1100s
    G_loss -> 1.9053003065129543, D_loss_real -> 0.23552483283577763, D_loss_fake -> 0.3951658665182743
    """

    output_dir = '../output/dcgan_test_pytorch'
    image_size = 64  # for 128 just add conv layer in both generator and discriminator and adjust shape
    batch_size = 128
    epochs = 200
    latent_dim = 100
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # Decide which device we want to run on
    dev = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)

    metadata_csv = pd.read_csv('../dataset/best_artworks/artists_good.csv')

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

    train_impressionist = PaintingsFolder(
        root='../dataset/best_artworks/images',
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
        root='../dataset/best_artworks/images',
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
        root='../dataset/best_artworks/images',
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
    noise_fn = lambda x: torch.normal(mean=0, std=1, size=(x, latent_dim), device=dev)
    gan = DCGAN(latent_dim, noise_fn, dataloader, batch_size=batch_size, device=dev)

    start = time()
    for i in range(epochs):
        print(f"Epoch {i + 1}; Elapsed time = {int(time() - start)}s")

        g_loss, (d_loss_real, d_loss_fake) = gan.train_epoch()

        print(f"G_loss -> {g_loss}, D_loss_real -> {d_loss_real}, D_loss_fake -> {d_loss_fake}\n")

        # save grid of 64 imgs for each epoch, to see generator progress
        images = gan.generate_samples(num=64)
        ims = vutils.make_grid(images, normalize=True)
        plt.axis("off")
        plt.title(f"Epoch {i + 1}")
        plt.imshow(np.transpose(ims, (1, 2, 0)))
        plt.savefig(f'{output_dir}/epoch{i + 1}.png')
