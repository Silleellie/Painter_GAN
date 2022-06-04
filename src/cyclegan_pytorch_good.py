import torch
import torch.nn as nn
from torch import optim
import torch.utils.data
import itertools

from abstract_gan import AB_GAN
from utils import device

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

        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(features_gen*4))
        
        layers.extend(self._default_block_upsample(features_gen*4, features_gen*2))
        layers.extend(self._default_block_upsample(features_gen*2, features_gen))

        layers.extend([nn.ReflectionPad2d(channels_img),
                       nn.Conv2d(features_gen, channels_img, 7, 1, 0),
                       nn.Tanh()])
        
        self.model = nn.Sequential(*layers)

    @staticmethod
    def _default_block_upsample(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1, output_padding=1):
        return [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=output_padding, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True)
        ]

    @staticmethod
    def _default_block_downsample(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1):
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
            Discriminator._default_block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=1, bias=False)
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

class CYCLEGAN(AB_GAN):
    def __init__(self, init_features: int = 64, 
                 lr_generator=2e-4, lr_discriminator=2e-4,
                 lambda_cycle=10.0, lambda_identity=5.0, num_res_blocks=6,
                 adversarial_criterion = nn.MSELoss(), cycle_criterion = nn.L1Loss(), identity_criterion = nn.L1Loss(),
                 lr_scheduler_class: torch.optim.lr_scheduler = torch.optim.lr_scheduler.LambdaLR):
        """
        suggested number of residual blocks equal to 6 for 128x128 images and equal to 9 for 256x256 or better quality images
        """

        gen_a2b = Generator(3, init_features, num_res_blocks).to(device)
        gen_b2a = Generator(3, init_features, num_res_blocks).to(device)
        dis_a = Discriminator(3, init_features).to(device)
        dis_b = Discriminator(3, init_features).to(device)

        super().__init__(generator_a2b=gen_a2b,
                         generator_b2a=gen_b2a,
                         discriminator_a=dis_a,
                         discriminator_b=dis_b,
                         optim_g=optim.Adam(itertools.chain(gen_a2b.parameters(), gen_b2a.parameters()),
                                            lr=lr_generator, betas=(0.5, 0.999)),
                         optim_d=optim.Adam(itertools.chain(dis_a.parameters(), dis_b.parameters()),
                                            lr=lr_discriminator, betas=(0.5, 0.999)),
                         lr_scheduler_class=lr_scheduler_class)

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

    def train_step(self, real_data_a, real_data_b):
        """
        Train both networks for one epoch and return the losses.
        Reference used for epoch training code: https://www.kaggle.com/code/lmyybh/pytorch-cyclegan
        """

        self.optim_g.zero_grad()

        ### GENERATOR TRAINING ###
    
        fake_B = self.generator_a2b(real_data_a)
        fake_A = self.generator_b2a(real_data_b)
        
        # identity loss
        loss_id_A = self.identity_criterion(fake_B, real_data_a)
        loss_id_B = self.identity_criterion(fake_A, real_data_b)
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
        loss_cycle_A = self.cycle_criterion(recov_A, real_data_a)
        loss_cycle_B = self.cycle_criterion(recov_B, real_data_b)
        loss_cycle = loss_cycle_A + loss_cycle_B

        loss_G = self.lambda_identity * loss_identity + loss_GAN + self.lambda_cycle * loss_cycle
        
        loss_G.backward()
        self.optim_g.step()

        self.optim_d.zero_grad()
        
        # DISCRIMINATOR A TRAINING
        
        pred_a = self.discriminator_a(real_data_a)
        pred_a_fake = self.discriminator_a(fake_A.detach())
        loss_real = self.adversarial_criterion(pred_a, torch.ones_like(pred_a))
        loss_fake = self.adversarial_criterion(pred_a_fake, torch.zeros_like(pred_a_fake))
        # divided by 2 as suggested in the CycleGan paper
        loss_D_A = (loss_real + loss_fake) / 2
        
        # DISCRIMINATOR B TRAINING
        
        pred_b = self.discriminator_b(real_data_b)
        pred_b_fake = self.discriminator_b(fake_B.detach())
        loss_real = self.adversarial_criterion(pred_b, torch.ones_like(pred_b))
        loss_fake = self.adversarial_criterion(pred_b_fake, torch.zeros_like(pred_b_fake))
        # divided by 2 as suggested in the CycleGan paper
        loss_D_B = (loss_real + loss_fake) / 2
        
        loss_D = loss_D_A + loss_D_B
        loss_D.backward()
        self.optim_d.step()

        loss_g = loss_G.item()
        loss_d = loss_D_A.item() + loss_D_B.item()

        return {"G_loss": loss_g, "D_loss": loss_d}

if __name__ == '__main__':
    """
    CYCLEGAN PAPER
    https://arxiv.org/abs/1703.10593
    """

    decay_epoch = 100
    epochs = 200

    def lr_decay_func(epoch): return 1 - max(0, epoch-decay_epoch)/(epochs-decay_epoch)
    
    gan = CYCLEGAN()
    gan.train(1, 128, epochs, scheduler_params={'lr_lambda': lr_decay_func}, create_progress_images=True, save_model_checkpoints=True)
