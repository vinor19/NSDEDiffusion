import torch
import torch.nn as nn
from Baysian_Layers import ConvCLTLayerDet
import math
from Models.UNetATT import SinusoidalPositionEmbeddings

#Helper function to make blocks used in UNetNSDE
class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvCLTLayerDet(in_channels, out_channels, 3, padding=1)
        self.act = nn.ReLU()
        self.conv2 = ConvCLTLayerDet(out_channels, out_channels, 3, padding=1)


    def forward(self, mu, var):
        mu1, var1 = self.conv1(mu,var)
        mu1 = self.act(mu1)
        var1 = self.act(var1)

        mu2, var2 = self.conv2(mu1,var1)
        mu2 = self.act(mu2)
        var2 = self.act(var2)
        return mu2,var2


# Simple class without much structure
class ConvNSDE(nn.Module):
    def __init__(self,channels):

        super().__init__()

        self.dconv_down = double_conv(channels, 64)
        self.conv_last = ConvCLTLayerDet(64, channels, 1)

    def forward(self, x, y):
        return self.conv_last(*self.dconv_down(x,y))


# UNet implemented using ConvCLTLayerDet for the convolutional layers, but still keeping upsampling and maxpooling
class UNetNSDE(nn.Module):

    def __init__(self, channels):
        super().__init__()
                
        self.dconv_down1 = double_conv(channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(64 + 128, 64)
        
        self.conv_last = ConvCLTLayerDet(64, channels, 1)
        
        
    def forward(self, x,y):
        mu_c1, var_c1 = self.dconv_down1(x,y)
        x = self.maxpool(mu_c1)
        y = self.maxpool(var_c1)

        mu_c2, var_c2 = self.dconv_down2(x,y)
        x = self.maxpool(mu_c2)
        y = self.maxpool(var_c2)

        mu_c3, var_c3 = self.dconv_down3(x,y)
        x = self.maxpool(mu_c3)
        y = self.maxpool(var_c3)   
        
        x, y = self.dconv_down4(x,y)
        
        x = self.upsample(x)        
        y = self.upsample(y)        
        x = torch.cat([x, mu_c3], dim=1)
        y = torch.cat([y, var_c3], dim=1)
        
        x,y = self.dconv_up3(x,y)
        x = self.upsample(x)        
        y = self.upsample(y)        
        x = torch.cat([x, mu_c2], dim=1)
        y = torch.cat([y, var_c2], dim=1)     


        x,y = self.dconv_up2(x,y)
        x = self.upsample(x)        
        y = self.upsample(y)        
        x = torch.cat([x, mu_c1], dim=1)
        y = torch.cat([y, var_c1], dim=1)    
        
        x,y = self.dconv_up1(x,y)
        
        out = self.conv_last(x,y)
        
        return out


#Helper function to make blocks used in UNet
def double_conv_func(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   

# Standard UNet structure
class UNetSimple(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv_func(n_class, 64)
        self.dconv_down2 = double_conv_func(64, 128)
        self.dconv_down3 = double_conv_func(128, 256)
        self.dconv_down4 = double_conv_func(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv_func(256 + 512, 256)
        self.dconv_up2 = double_conv_func(128 + 256, 128)
        self.dconv_up1 = double_conv_func(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out