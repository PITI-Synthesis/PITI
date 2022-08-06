import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from .nn import mean_flat
from . import dist_util
import functools

class AdversarialLoss(nn.Module):
    def __init__(self,  gan_type='WGAN_GP', gan_k=1, 
        lr_dis=1e-5 ):

        super(AdversarialLoss, self).__init__()
       
        self.gan_type = gan_type
        self.gan_k = gan_k
 
        model = NLayerDiscriminator().to(dist_util.dev()) 
        self.discriminator =  DDP(
                model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
 
        if (gan_type in ['WGAN_GP', 'GAN']):
            self.optimizer = optim.Adam(
                self.discriminator.parameters(),
                 lr=lr_dis
            )
   
    def forward(self, fake, real):
        fake_detach = fake.detach()
        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            d_fake = self.discriminator(fake_detach)
            d_real = self.discriminator(real)
            if (self.gan_type.find('WGAN') >= 0):
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand(real.size(0), 1, 1, 1).to(dist_util.dev())
                    epsilon = epsilon.expand(real.size())
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty
 
            # print('d loss:', loss_d)
            # Discriminator update
            loss_d.backward()
            self.optimizer.step()

        d_fake_for_g = self.discriminator(fake)
        if (self.gan_type.find('WGAN') >= 0):
            loss_g = -d_fake_for_g 
    
        # Generator loss
        return  mean_flat(loss_g)



def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=True)


def conv7x7(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, 
                     stride=stride, padding=3, bias=True)


class Discriminator(nn.Module):
    def __init__(self, ):
        super(Discriminator, self).__init__()
        self.conv1 = conv7x7(3, 32)
        self.norm1 = nn.InstanceNorm2d(32, affine=True)
        self.LReLU1 = nn.LeakyReLU(0.2)

        self.conv2 = conv3x3(32, 32, 2)
        self.norm2 = nn.InstanceNorm2d(32, affine=True)
        self.LReLU2 = nn.LeakyReLU(0.2)

        self.conv3 = conv3x3(32, 64)
        self.norm3 = nn.InstanceNorm2d(64, affine=True)
        self.LReLU3 = nn.LeakyReLU(0.2)

        self.conv4 = conv3x3(64, 64, 2)
        self.norm4 = nn.InstanceNorm2d(64, affine=True)
        self.LReLU4 = nn.LeakyReLU(0.2)

        self.conv5 = conv3x3(64, 128)
        self.norm5 = nn.InstanceNorm2d(128, affine=True)
        self.LReLU5 = nn.LeakyReLU(0.2)

        self.conv6 = conv3x3(128, 128, 2)
        self.norm6 = nn.InstanceNorm2d(128, affine=True)
        self.LReLU6 = nn.LeakyReLU(0.2)

        self.conv7 = conv3x3(128, 256)
        self.norm7 = nn.InstanceNorm2d(256, affine=True)
        self.LReLU7 = nn.LeakyReLU(0.2)

        self.conv8 = conv3x3(256, 256, 2)
        self.norm8 = nn.InstanceNorm2d(256, affine=True)
        self.LReLU8 = nn.LeakyReLU(0.2)

        self.conv9 = conv3x3(256, 512)
        self.norm9 = nn.InstanceNorm2d(512, affine=True)
        self.LReLU9 = nn.LeakyReLU(0.2)

        self.conv10 = conv3x3(512, 512, 2)
        self.norm10 = nn.InstanceNorm2d(512, affine=True)
        self.LReLU10 = nn.LeakyReLU(0.2)

        self.conv11 = conv3x3(512, 32)
        self.norm11 = nn.InstanceNorm2d(32, affine=True)
        self.LReLU11 = nn.LeakyReLU(0.2)
        self.conv12 = conv3x3(32, 1)
 
        
    def forward(self, x):
        x = self.LReLU1(self.norm1(self.conv1(x)))
        x = self.LReLU2(self.norm2(self.conv2(x)))
        x = self.LReLU3(self.norm3(self.conv3(x)))
        x = self.LReLU4(self.norm4(self.conv4(x)))
        x = self.LReLU5(self.norm5(self.conv5(x)))
        x = self.LReLU6(self.norm6(self.conv6(x)))
        x = self.LReLU7(self.norm7(self.conv7(x)))
        x = self.LReLU8(self.norm8(self.conv8(x)))
        x = self.LReLU9(self.norm9(self.conv9(x)))
        x = self.LReLU10(self.norm10(self.conv10(x)))
        
        x = self.LReLU11(self.norm11(self.conv11(x)))
        x = self.conv12(x)
 
        return x        



def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=3, ndf=64, n_layers=3 ):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        norm_layer = get_norm_layer(norm_type='instance')
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)