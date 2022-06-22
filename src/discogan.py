import torch
import torch.nn as nn
from torch import optim
import torch.utils.data
import itertools

from src.abstract_gan import ABGAN
from src.utils import device

class Generator(nn.Module):
    def __init__(self, channels_img, features_gen):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(channels_img, features_gen, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2 , inplace=True),
            Generator._default_block_downsample(features_gen, features_gen * 2, 4, 2, 1),
            Generator._default_block_downsample(features_gen * 2, features_gen * 4, 4, 2, 1),
            Generator._default_block_downsample(features_gen * 4, features_gen * 8, 4, 2, 1),
        )

        self.upsample = nn.Sequential(
            Generator._default_block_upsample(features_gen * 8, features_gen * 4, 4, 2, 1),
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
            Discriminator._default_block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
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

class DISCOGAN(ABGAN):
    def __init__(self, init_features: int = 64,
                 lr_generator=0.0002, lr_discriminator=0.0002, weight_decay: int = 0.00001,
                 reconstruction_criterion=nn.MSELoss(), gan_criterion=nn.BCELoss()): 

        gen_a2b = Generator(3, init_features).to(device)
        gen_b2a = Generator(3, init_features).to(device)

        dis_a = Discriminator(3, init_features).to(device)
        dis_b = Discriminator(3, init_features).to(device)
        
        optim_g = optim.Adam(itertools.chain(gen_a2b.parameters(), gen_b2a.parameters()),
                             lr=lr_generator, betas=(0.5, 0.999), weight_decay=weight_decay)
        optim_d = optim.Adam(itertools.chain(dis_a.parameters(), dis_b.parameters()),
                             lr=lr_discriminator, betas=(0.5, 0.999), weight_decay=weight_decay)

        super().__init__(generator_a2b=gen_a2b,
                         generator_b2a=gen_b2a,
                         discriminator_a=dis_a,
                         discriminator_b=dis_b,
                         optim_g=optim_g,
                         optim_d=optim_d)

        # how realistic a generated image is in domain B
        self.gan_criterion = gan_criterion

        # how well the original input is reconstructed after a sequence of 2 generations
        self.reconstruction_criterion = reconstruction_criterion

    def train_step(self, real_data_a, real_data_b):

        ### GENERATOR TRAINING ###

        self.generator_a2b.train()
        self.generator_b2a.train()

        self.optim_g.zero_grad()

        AB = self.generator_a2b(real_data_a)
        BA = self.generator_b2a(real_data_b)

        ABA = self.generator_b2a(AB)
        BAB = self.generator_a2b(BA)

        pred_real_ba = self.discriminator_a(BA)
        pred_real_ab = self.discriminator_b(AB)
        GAN_loss = self.gan_criterion(pred_real_ba, torch.ones_like(pred_real_ba).to(device)) + self.gan_criterion(pred_real_ab, torch.ones_like(pred_real_ab).to(device))

        # Reconstruction Loss
        recon_loss_A = self.reconstruction_criterion(ABA, real_data_a)
        recon_loss_B = self.reconstruction_criterion(BAB, real_data_b)
        recon_loss = recon_loss_A + recon_loss_B

        loss_g = recon_loss + GAN_loss

        loss_g.backward()
        self.optim_g.step()

        ### DISCRIMINATOR TRAINING ###

        self.optim_d.zero_grad()

        pred_a = self.discriminator_a(real_data_a)
        pred_b = self.discriminator_b(real_data_b)
        pred_a_fake = self.discriminator_a(BA.detach())
        pred_b_fake = self.discriminator_b(AB.detach())
        loss_d_a = self.gan_criterion(pred_a, torch.ones_like(pred_a).to(device)) * 0.5 + self.gan_criterion(pred_a_fake, torch.zeros_like(pred_a_fake).to(device)) * 0.5
        loss_d_b = self.gan_criterion(pred_b, torch.ones_like(pred_b).to(device)) * 0.5 + self.gan_criterion(pred_b_fake, torch.zeros_like(pred_b_fake).to(device)) * 0.5
        loss_d = loss_d_a + loss_d_b

        loss_d.backward()
        self.optim_d.step()

        loss_d = loss_d.item()
        loss_g = loss_g.item()

        return {"G_loss": loss_g, "D_loss": loss_d}

if __name__ == '__main__':
    gan = DISCOGAN()
    gan.train_with_default_dataset(64, 64, 10, False, False)
            