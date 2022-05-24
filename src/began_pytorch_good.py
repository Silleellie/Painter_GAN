import os
import random
import re
import shutil
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset

from src.utils import PaintingsFolder


class Decoder(nn.Module):
    def __init__(self, latent_dimension, num_filters):
        super(Decoder, self).__init__()
        self.num_filters = num_filters
        self.h = latent_dimension  # latent dimension is called 'h' in paper

        self._init_modules()

    def _init_modules(self):
        self.elu = nn.ELU(inplace=True)
        self.tanh = nn.Tanh()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

        self.h0 = nn.Linear(self.h, self.num_filters * 8 * 8)
        self.conv1 = nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1)
        self.conv2 = nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1)

        self.conv3 = nn.Conv2d(2 * self.num_filters, self.num_filters, 3, 1, 1)
        self.conv4 = nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1)

        # self.conv5 = nn.Conv2d(2 * self.num_filters, self.num_filters, 3, 1, 1)
        # self.conv6 = nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1)

        self.conv7 = nn.Conv2d(2 * self.num_filters, self.num_filters, 3, 1, 1)
        self.conv8 = nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1)

        self.conv9 = nn.Conv2d(self.num_filters, 3, 3, 1, 1)

    def weights_init(self):
        for m in self._modules:
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                m.weight.data.normal_(0.0, 0.2)
                if m.bias.data is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        h0 = self.h0(input)

        # reshape
        h0 = h0.view((input.shape[0], self.num_filters, 8, 8))

        x = self.elu(self.conv1(h0))
        x = self.elu(self.conv2(x))
        x = self.up(x)

        # upsample and inject h0
        upsampled_h = self.up(h0)
        x = torch.cat([x, upsampled_h], dim=1)

        x = self.elu(self.conv3(x))
        x = self.elu(self.conv4(x))
        x = self.up(x)

        # upsample and inject
        upsampled_h = self.up(upsampled_h)
        x = torch.cat([x, upsampled_h], dim=1)

        # x = self.elu(self.conv5(x))
        # x = self.elu(self.conv6(x))
        # x = self.up(x)
        #
        # # upsample and inject
        # upsampled_h = self.up(upsampled_h)
        # x = torch.cat([x, upsampled_h], dim=1)

        x = self.elu(self.conv7(x))
        x = self.elu(self.conv8(x))

        x = self.elu(self.conv9(x))
        x = self.tanh(x)
        return x


class Encoder(nn.Module):
    def __init__(self, latent_dimension, num_filters):
        super(Encoder, self).__init__()
        self.num_filters = num_filters
        self.h = latent_dimension

        self._init_modules()

    def _init_modules(self):
        self.elu = nn.ELU(inplace=True)

        # last conv of each layer will take care of subsampling with stride=2

        self.conv0 = nn.Conv2d(3, self.num_filters, 3, 1, 1)
        self.conv1 = nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1)
        self.conv2 = nn.Conv2d(self.num_filters, 2 * self.num_filters, 3, 2, 1)

        self.conv3 = nn.Conv2d(2 * self.num_filters, 2 * self.num_filters, 3, 1, 1)
        self.conv4 = nn.Conv2d(2 * self.num_filters, 3 * self.num_filters, 3, 2, 1)

        # self.conv5 = nn.Conv2d(3 * self.num_filters, 3 * self.num_filters, 3, 1, 1)
        # self.conv6 = nn.Conv2d(3 * self.num_filters, 3 * self.num_filters, 3, 1, 1)
        # self.layer3_subsampling = nn.Conv2d(3 * self.num_filters, 4 * self.num_filters, 3, 2, 1)

        self.conv7 = nn.Conv2d(3 * self.num_filters, 3 * self.num_filters, 3, 1, 1)
        self.conv8 = nn.Conv2d(3 * self.num_filters, 3 * self.num_filters, 3, 1, 1)

        # output of encoder will be input of encoder which expects input of size 'h'
        self.fc = nn.Linear(8 * 8 * 3 * self.num_filters, self.h)

    def weights_init(self):
        for m in self._modules:
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                m.weight.data.normal_(0.0, 0.2)
                if m.bias.data is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        x = self.elu(self.conv0(input))
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))

        x = self.elu(self.conv3(x))
        x = self.elu(self.conv4(x))

        # x = self.elu(self.conv5(x))
        # x = self.elu(self.conv6(x))
        # x = self.elu(self.layer3_subsampling(x))

        x = self.elu(self.conv7(x))
        x = self.elu(self.conv8(x))

        # reshape
        x = x.view((input.shape[0], 3 * self.num_filters * 8 * 8))
        x = self.fc(x)

        return x


# alias for the decoder, since generator has same architecture
class Generator(nn.Module):
    def __init__(self, latent_dimension=100, num_filters=64):
        super(Generator, self).__init__()
        self.dec = Decoder(latent_dimension=latent_dimension, num_filters=num_filters)

    def weights_init(self):
        self.dec.weights_init()

    def forward(self, input):
        out = self.dec(input)

        return out


class Discriminator(nn.Module):
    def __init__(self, latent_dimension=100, num_filters=64):
        super(Discriminator, self).__init__()
        self.enc = Encoder(latent_dimension=latent_dimension, num_filters=num_filters)
        self.dec = Decoder(latent_dimension=latent_dimension, num_filters=num_filters)

    def weights_init(self):
        self.enc.weights_init()
        self.dec.weights_init()

    def forward(self, input):
        out = self.enc(input)
        out = self.dec(out)

        return out


class BEGAN:
    def __init__(self, latent_dim, num_filters, noise_fn, dataloader, total_epochs,
                 batch_size=32, lr_d=0.0004, lr_g=0.0004, device: torch.device = 'cpu'):
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
        self.generator = Generator(latent_dim, num_filters).to(device)
        self.generator.weights_init()

        self.discriminator = Discriminator(latent_dim, num_filters).to(device)
        self.discriminator.weights_init()

        self.noise_fn = noise_fn
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.device = device

        self.criterion = nn.L1Loss()
        self.optim_d = optim.Adam(self.discriminator.parameters(),
                                  lr=lr_d, betas=(0.5, 0.999))
        self.optim_g = optim.Adam(self.generator.parameters(),
                                  lr=lr_g, betas=(0.5, 0.999))

        self.gamma = 0.5  # controls diversity of generated images
        self.lambda_k = 0.001  # used in paper
        self.Kt = 0.0  # starts at 0

        self.global_iter = 0
        self.lr_step_size = len(self.dataloader.dataset) // self.batch_size * self.total_epochs // 8

        self.scheduler_d = optim.lr_scheduler.StepLR(self.optim_d, step_size=1, gamma=0.5)
        self.scheduler_g = optim.lr_scheduler.StepLR(self.optim_g, step_size=1, gamma=0.5)

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

        latent_vec = self.noise_fn(current_batch_size)
        generated = self.generator(latent_vec)
        classifications = self.discriminator(generated)
        loss = self.criterion(generated, classifications)  # l1_loss

        self.optim_g.zero_grad()
        loss.backward()
        self.optim_g.step()

        return loss

    def train_step_discriminator(self, real_samples, current_batch_size):
        """Train the discriminator one step and return the losses."""
        pred_real = self.discriminator(real_samples)
        loss_real = self.criterion(pred_real, real_samples)  # l1_loss

        # generated samples
        latent_vec = self.noise_fn(current_batch_size)
        fake_samples = self.generator(latent_vec)
        pred_fake = self.discriminator(fake_samples.detach())
        loss_fake = self.criterion(pred_fake, fake_samples)  # l1_loss

        # combine two losses
        loss = loss_real - self.Kt * loss_fake

        self.optim_d.zero_grad()
        loss.backward()
        self.optim_d.step()

        return loss, loss_real, loss_fake

    def train_epoch(self):
        loss_g_running, loss_d_running, M_running = 0, 0, 0
        for batch, (real_samples, _) in enumerate(self.dataloader):
            self.global_iter += 1

            current_batch_size = real_samples.size(0)

            real_samples = real_samples.to(self.device)

            # ----------
            #  Training generator
            # ----------
            loss_g = self.train_step_generator(current_batch_size)

            # ----------
            #  Training discriminator
            # ----------
            loss_d, loss_d_real, loss_d_fake = self.train_step_discriminator(real_samples, current_batch_size)

            # ----------------
            # Update weights
            # ----------------
            # Kt update
            balance = (self.gamma * loss_d_real - loss_d_fake)
            self.Kt = max(min(self.Kt + self.lambda_k * balance.item(), 1.0), 0.0)

            # Update convergence metric
            M = (loss_d_real + torch.abs(balance)).item()

            # ----------------
            # Update metrics for logging
            # ----------------
            loss_d_running += loss_d.item()
            loss_g_running += loss_g.item()
            M_running += M

            if self.global_iter % self.lr_step_size == 0:
                self.scheduler_g.step()
                self.scheduler_d.step()

        n_batches = len(self.dataloader)
        loss_g_running /= n_batches
        loss_d_running /= n_batches
        M_running /= n_batches

        return loss_g_running, loss_d_running, M_running, self.Kt


def clean_dataset(resized_images_dir):
    for filename in os.listdir(resized_images_dir):

        if os.path.isfile(os.path.join(resized_images_dir, filename)):
            if re.match(r'Albrecht_D(?=.+rer_(\d+))', filename):

                painting_number = re.findall(r'Albrecht_D(?=.+rer_(\d+)\.jpg)', filename)[0]

                old_fullpath = os.path.join(resized_images_dir, filename)

                filename = f'Albrecht_Dürer_{painting_number}.jpg'
                new_fullpath = os.path.join(resized_images_dir, filename)

                if not os.path.isfile(new_fullpath):
                    os.rename(old_fullpath, new_fullpath)

            artist_name = re.findall(r'(.*?[_.*?]*)(?=_\d+)', filename, re.UNICODE)[0]

            Path(os.path.join(resized_images_dir, artist_name)).mkdir(parents=True, exist_ok=True)
            shutil.move(os.path.join(resized_images_dir, filename), os.path.join(resized_images_dir,
                                                                                 artist_name,
                                                                                 filename))


if __name__ == '__main__':
    """
    BEGAN PAPER
    https://arxiv.org/pdf/1703.10717.pdf
    
    IMPROVED BEGAN PAPER
    https://wlouyang.github.io/Papers/iBegan.pdf
    """
    output_dir = "../output/began_test_pytorch"

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)

    resized_images_dir = '../dataset/best_artworks/resized/resized'
    image_size = 32
    batch_size = 16
    epochs = 20
    latent_dim = 64
    num_filters = 64
    lr_discriminator = 0.0001
    lr_generator = 0.0001

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
    noise_fn = lambda x: torch.normal(mean=0, std=1, size=(x, latent_dim), device=device)


    gan = BEGAN(latent_dim, num_filters, noise_fn, dataloader, batch_size=batch_size, total_epochs=epochs,
                device=device, lr_d=lr_discriminator, lr_g=lr_generator)

    print("Started training...")
    for i in range(epochs):
        g_loss, d_loss, M, Kt = gan.train_epoch()

        print(
            "[Epoch %d/%d] [G loss: %f] [D loss: %f]  -- M: %f, k: %f"
            % (i + 1, epochs, g_loss, d_loss, M, Kt)
        )

        # save grid of 64 imgs for each epoch, to see generator progress
        images = gan.generate_samples(num=64)
        ims = vutils.make_grid(images, normalize=True)
        plt.axis("off")
        plt.title(f"Epoch {i + 1}")
        plt.imshow(np.transpose(ims, (1, 2, 0)))
        plt.savefig(f'{output_dir}/epoch{i + 1}.png')
