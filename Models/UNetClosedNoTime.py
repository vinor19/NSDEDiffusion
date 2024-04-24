import torch
import torch.nn as nn
from Baysian_Layers import ConvCLTLayerDet

#Helper function to make blocks used in UNetNSDE
class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvCLTLayerDet(in_channels, out_channels, 3, padding=1)
        self.act = nn.ReLU()
        self.conv2 = ConvCLTLayerDet(out_channels, out_channels, 3, padding=1)

    def forward(self, mu, var):
        mu1, var1 = self.conv1(mu,var)
        mu2, var2 = self.conv2(mu1,var1)
        return mu2,var2

# Simple class without much structure
class ConvNSDE(nn.Module):
    def __init__(self,channels):

        super().__init__()

        self.dconv_down = double_conv(channels, 64)
        self.conv_last = ConvCLTLayerDet(64, channels, 1)

    def forward(self, x, y):
        return self.conv_last(*self.dconv_down(x,y))

class UNetNSDE2(nn.Module):

    def __init__(self, channels):
        super().__init__()
                
        self.dconv_down1 = double_conv(channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)

        self.decoder_block1 = double_conv(512,512)
        self.decoder_block2 = double_conv(512,512)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(64 + 128, 64)
        
        self.conv_last = ConvCLTLayerDet(64, channels, 1, isoutput=True)
        
        
    def forward(self, x, y):
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

        x, y = self.decoder_block1(x,y)
        x, y = self.decoder_block2(x,y)
        
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

# UNet implemented using ConvCLTLayerDet for the output
class UNetNSDE(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.dconv_down1 = double_conv_func(n_class, 64, 0)
        self.dconv_down2 = double_conv_func(64, 128, 0)
        self.dconv_down3 = double_conv_func(128, 256, 0)
        self.dconv_down4 = double_conv_func(256, 512, 0)        
        self.maxpool = nn.MaxPool2d(2)

        self.decoder_block1 = double_conv_func(512,512, 0)
        self.decoder_block2 = double_conv_func(512,512, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv_func(256 + 512, 256, 0)
        self.dconv_up2 = double_conv_func(128 + 256, 128, 0)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = ConvCLTLayerDet(64, n_class, 1, isoutput=True)
        
        
    def forward(self, x, ts):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.decoder_block1(x)
        x = self.decoder_block1(x)

        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   

        y = torch.ones_like(x)*1e-6
        x, y = self.dconv_up1(x, y)
        
        out, var = self.conv_last(x, y)
        
        return out, var

# UNet implemented using ConvCLTLayerDet for decoder blocks
class UNetNSDEMid(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.dconv_down1 = double_conv_func(channels, 64, 0)
        self.dconv_down2 = double_conv_func(64, 128, 0)
        self.dconv_down3 = double_conv_func(128, 256, 0)
        self.dconv_down4 = double_conv_func(256, 512, 0)        
        self.maxpool = nn.MaxPool2d(2)

        self.decoder_block1 = double_conv(512,512)
        self.decoder_block2 = double_conv(512,512)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(64 + 128, 64)
        
        self.conv_last = ConvCLTLayerDet(64, channels, 1, isoutput=True)
        
        
    def forward(self, x, ts):
        mu1 = self.dconv_down1(x)
        x = self.maxpool(mu1)

        mu2 = self.dconv_down2(x)
        x = self.maxpool(mu2)
        
        mu3 = self.dconv_down3(x)
        x = self.maxpool(mu3)   
        
        x = self.dconv_down4(x)
        
        y = torch.ones_like(x)*1e-6
        x, y = self.decoder_block1(x, y)
        x, y = self.decoder_block1(x, y)

        x = self.upsample(x)        
        y = self.upsample(y)        
        x = torch.cat([x, mu3], dim=1)
        y = torch.cat([y, torch.ones_like(mu3)*1e-6], dim=1)
        
        x,y = self.dconv_up3(x,y)
        x = self.upsample(x)        
        y = self.upsample(y)        
        x = torch.cat([x, mu2], dim=1)
        y = torch.cat([y, torch.ones_like(mu2)*1e-6], dim=1)     


        x,y = self.dconv_up2(x,y)
        x = self.upsample(x)        
        y = self.upsample(y)        
        x = torch.cat([x, mu1], dim=1)
        y = torch.cat([y, torch.ones_like(mu1)*1e-6], dim=1)    
        
        x,y = self.dconv_up1(x,y)
        
        out, var = self.conv_last(x,y)
        
        return out, var
    
#Helper function to make blocks used in UNet
def double_conv_func(in_channels, out_channels, p=0.1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=p),
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
        self.decoder_block1 = double_conv_func(512,512)
        self.decoder_block2 = double_conv_func(512,512)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv_func(256 + 512, 256)
        self.dconv_up2 = double_conv_func(128 + 256, 128)
        self.dconv_up1 = double_conv_func(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x,ts):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.decoder_block1(x)
        x = self.decoder_block1(x)

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