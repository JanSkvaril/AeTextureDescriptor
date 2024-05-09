import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as  np


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, transposed=False):
        super(UpConvBlock, self).__init__()
        if transposed:
            up = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        else:
            up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            )
        self.conv = nn.Sequential(
            up,
            nn.ReLU( inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),

        )

    def forward(self, x):
        x = self.conv(x)
        return x

class CNN_Encoder(nn.Module):
    def __init__(self, output_size, input_size=(128, 128), dropout=0.5):
        super(CNN_Encoder, self).__init__()

        self.input_size = input_size

        channels = np.array([32, 64, 128, 256, 512])
        self.layers = [
            nn.Conv2d(1, channels[0], 3,2,1),
            nn.ReLU( inplace=True)
            ]

        down_layers = 5
        for i in range(down_layers-1):
            self.layers.append(nn.Conv2d(channels[i], channels[i+1], 3,2,1))
            self.layers.append(nn.ReLU( inplace=True))
            self.layers.append(nn.BatchNorm2d(channels[i+1]))
            self.layers.append(nn.Dropout2d(dropout))
            self.layers.append(nn.Conv2d(channels[i+1], channels[i+1], 3,1,1))
            self.layers.append(nn.ReLU( inplace=True))
            self.layers.append(nn.BatchNorm2d(channels[i+1]))
            
      

        self.conv = nn.Sequential(*self.layers)


        self.last_conv_size = (channels[down_layers-1], input_size[0]//(2**down_layers), input_size[1]//(2**down_layers))
        self.flatten = nn.Flatten()

        self.linear = nn.Sequential(
            nn.Linear(self.last_conv_size[0] * self.last_conv_size[1] * self.last_conv_size[2], output_size),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(0.2),
        )


    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)
        return F.sigmoid(x)

class CNN_Decoder(nn.Module):
    def __init__(self, input_size, latent_dim=1024, dropout=0.0, transposed=False):
        super(CNN_Decoder, self).__init__()
        
        self.input_size = input_size

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, input_size[0]  * input_size[1]  * input_size[2]),
            nn.BatchNorm1d(input_size[0]  * input_size[1]  * input_size[2]),
            nn.ReLU( inplace=True),
        )

        up_layers = 5
        self.layers = []
        channels = np.array([512, 256, 128, 64, 32])
        for i in range(up_layers-1):
            self.layers.append(UpConvBlock(channels[i], channels[i+1],transposed=transposed))
            self.layers.append(nn.ReLU( inplace=True))
            self.layers.append(nn.BatchNorm2d(channels[i+1]))
            self.layers.append(nn.Dropout2d(dropout))
            self.layers.append(nn.Conv2d(channels[i+1], channels[i+1], 3,1,1))
            self.layers.append(nn.ReLU( inplace=True))
            self.layers.append(nn.BatchNorm2d(channels[i+1]))

        self.layers.append(UpConvBlock(channels[up_layers-1], 8))
        self.layers.append(nn.ReLU( inplace=True))
        
        self.layers.append(nn.Conv2d(8, 1, 3,1,1))

        self.conv = nn.Sequential(*self.layers)


    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.input_size[0], self.input_size[1], self.input_size[2])
        x = self.conv(x)
        return x
    



class AutoEncoder(nn.Module):
    def __init__(self, latent_dim, input_size=(128, 128), dropout=0.5, transposed=False):
        super(AutoEncoder, self).__init__()
        self.encoder = CNN_Encoder(latent_dim, input_size, dropout=dropout)
        self.decoder = CNN_Decoder(input_size=self.encoder.last_conv_size,latent_dim=latent_dim, dropout=dropout, transposed=transposed)
        
    def forward(self, x):
        x = self.encoder(x)
        self.z = x
        x = self.decoder(x)

        return F.sigmoid(x)
    

class VAE(nn.Module):
    def __init__(self, latent_dim, input_size=(128, 128), dropout=0.5,transposed=False):
        super(VAE, self).__init__()
        self.encoder = CNN_Encoder(latent_dim, input_size, dropout=dropout)
        self.decoder = CNN_Decoder(input_size=self.encoder.last_conv_size,latent_dim=latent_dim, dropout=dropout,transposed=transposed)
        self.std_nn = nn.Linear(latent_dim, latent_dim)
        self.mean_nn = nn.Linear(latent_dim, latent_dim)
        
        self.latent_dim = latent_dim
        self.std_freezed = False
    def forward(self, x):
        x = self.encoder(x)
        #x = F.sigmoid(x) 
       # x= F.relu(x)
        self.mean = self.mean_nn(x)
        if not self.std_freezed:
            self.std = self.std_nn(x)
        else:
            self.std = Variable(torch.ones(self.mean.size()).cuda(), requires_grad=False)
        self.z = x
        self.sample_z = self.mean + self.std * Variable(torch.randn(self.mean.size()).cuda(), requires_grad=True)

        x = self.decoder(self.sample_z)

        return F.sigmoid(x)
    def FreezeStd(self):

        self.std_freezed = True
    def UnfreezeStd(self):

        self.std_freezed = False

    def KL_loss(self):
        return - 0.5 * torch.sum(1+ self.std - self.mean.pow(2) - self.std.exp())


