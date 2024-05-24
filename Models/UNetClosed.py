import torch
import torch.nn as nn
from Baysian_Layers import ConvCLTLayerDet
from Models.UNetATT import SinusoidalPositionEmbeddings

#Helper function to make blocks used in UNetCLT
class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb):
        super().__init__()
        self.dense = nn.Linear(time_emb, out_channels)
        self.conv1 = ConvCLTLayerDet(in_channels, out_channels, 3, padding=1)
        self.act = nn.ReLU(inplace=False)
        self.conv2 = ConvCLTLayerDet(out_channels, out_channels, 3, padding=1)

    def forward(self, mu, var, t):
        mu1, var1 = self.conv1(mu,var)
        emb = self.dense(self.act(t))[:, :, None, None]
        mu2 = mu1 + emb
        # var2 = var1 + emb
        mu3, var3 = self.conv2(mu2,var1)
        return mu3, var3


#Helper function to make blocks used in UNet
class double_conv_no_clt(nn.Module):
    def __init__(self,in_channels, out_channels, time_emb, p=0.1):
        super().__init__()
        self.dense = nn.Linear(time_emb, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.drop = nn.Dropout(p=p)
        self.act = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, mu, t):
        mu1 = self.act(self.conv1(mu))
        emb = self.dense(self.act(t))[:, :, None, None]
        mu2 = mu1 + emb
        mu3 = self.drop(mu2)
        mu4 = self.act(self.conv2(mu3))
        return mu4

# Simple class without much structure
class ConvCLT(nn.Module):
    def __init__(self,channels):

        super().__init__()

        self.dconv_down = double_conv(channels, 64)
        self.conv_last = ConvCLTLayerDet(64, channels, 1)

    def forward(self, x, y):
        return self.conv_last(*self.dconv_down(x,y))

class UNetCLT2(nn.Module):

    def __init__(self, base_channels, channels, time_multiple = 2):
        super().__init__()

        time_emb_dims_exp = base_channels * time_multiple
        self.time_embeddings = SinusoidalPositionEmbeddings(time_emb_dims=base_channels, time_emb_dims_exp=time_emb_dims_exp)
        self.dconv_down1 = double_conv(channels, 64, time_emb_dims_exp)
        self.dconv_down2 = double_conv(64, 128, time_emb_dims_exp)
        self.dconv_down3 = double_conv(128, 256, time_emb_dims_exp)
        self.dconv_down4 = double_conv(256, 512, time_emb_dims_exp)        

        self.maxpool = nn.MaxPool2d(2)

        self.decoder_block1 = double_conv(512,512, time_emb_dims_exp)
        self.decoder_block2 = double_conv(512,512, time_emb_dims_exp)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256, time_emb_dims_exp)
        self.dconv_up2 = double_conv(128 + 256, 128, time_emb_dims_exp)
        self.dconv_up1 = double_conv(64 + 128, 64, time_emb_dims_exp)
        
        self.conv_last = ConvCLTLayerDet(64, channels, 1, isoutput=True)
        
        
    def forward(self, x, y, ts):
        time_emb = self.time_embeddings(ts)

        mu_c1, var_c1 = self.dconv_down1(x,y,time_emb)
        x = self.maxpool(mu_c1)
        y = self.maxpool(var_c1)

        mu_c2, var_c2 = self.dconv_down2(x,y,time_emb)
        x = self.maxpool(mu_c2)
        y = self.maxpool(var_c2)

        mu_c3, var_c3 = self.dconv_down3(x,y,time_emb)
        x = self.maxpool(mu_c3)
        y = self.maxpool(var_c3)   
        
        x, y = self.dconv_down4(x,y,time_emb)

        x, y = self.decoder_block1(x,y,time_emb)
        x, y = self.decoder_block2(x,y,time_emb)
        
        x = self.upsample(x)        
        y = self.upsample(y)        
        x = torch.cat([x, mu_c3], dim=1)
        y = torch.cat([y, var_c3], dim=1)
        
        x,y = self.dconv_up3(x,y,time_emb)
        x = self.upsample(x)        
        y = self.upsample(y)        
        x = torch.cat([x, mu_c2], dim=1)
        y = torch.cat([y, var_c2], dim=1)     


        x,y = self.dconv_up2(x,y,time_emb)
        x = self.upsample(x)        
        y = self.upsample(y)        
        x = torch.cat([x, mu_c1], dim=1)
        y = torch.cat([y, var_c1], dim=1)    
        
        x,y = self.dconv_up1(x,y,time_emb)
        
        out = self.conv_last(x,y)


        
        return out

# UNet implemented using ConvCLTLayerDet for the output
class UNetCLT(nn.Module):

    def __init__(self, n_class, base_channels, time_multiple = 2):
        super().__init__()
        time_emb_dims_exp = base_channels * time_multiple
        self.time_embeddings = SinusoidalPositionEmbeddings(time_emb_dims=base_channels, time_emb_dims_exp=time_emb_dims_exp)
        self.dconv_down1 = double_conv_no_clt(n_class, 64, 0, time_emb_dims_exp)
        self.dconv_down2 = double_conv_no_clt(64, 128, 0, time_emb_dims_exp)
        self.dconv_down3 = double_conv_no_clt(128, 256, 0, time_emb_dims_exp)
        self.dconv_down4 = double_conv_no_clt(256, 512, 0, time_emb_dims_exp)        
        self.maxpool = nn.MaxPool2d(2)

        self.decoder_block1 = double_conv_no_clt(512,512, 0, time_emb_dims_exp)
        self.decoder_block2 = double_conv_no_clt(512,512, 0, time_emb_dims_exp)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv_no_clt(256 + 512, 256, 0, time_emb_dims_exp)
        self.dconv_up2 = double_conv_no_clt(128 + 256, 128, 0, time_emb_dims_exp)
        self.dconv_up1 = double_conv(128 + 64, 64, time_emb_dims_exp)
        
        self.conv_last = ConvCLTLayerDet(64, n_class, 1, isoutput=True)
        
        
    def forward(self, x, ts):
        time_emb = self.time_embeddings(ts)
        conv1 = self.dconv_down1(x, time_emb)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x, time_emb)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x, time_emb)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x, time_emb)
        
        x = self.decoder_block1(x, time_emb)
        x = self.decoder_block1(x, time_emb)

        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x, time_emb)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x, time_emb)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   

        y = torch.ones_like(x)*1e-6
        x, y = self.dconv_up1(x, y, time_emb)
        
        out, var = self.conv_last(x, y)
        
        return out, var

# UNet implemented using ConvCLTLayerDet for decoder blocks
class UNetCLTMid(nn.Module):

    def __init__(self, channels, base_channels, time_multiple = 2):
        super().__init__()
        time_emb_dims_exp = base_channels * time_multiple
        self.time_embeddings = SinusoidalPositionEmbeddings(time_emb_dims=base_channels, time_emb_dims_exp=time_emb_dims_exp)
        self.dconv_down1 = double_conv_no_clt(channels, 64, time_emb_dims_exp, 0)
        self.dconv_down2 = double_conv_no_clt(64, 128, time_emb_dims_exp, 0)
        self.dconv_down3 = double_conv_no_clt(128, 256, time_emb_dims_exp, 0)
        self.dconv_down4 = double_conv_no_clt(256, 512, time_emb_dims_exp, 0)        
        self.maxpool = nn.MaxPool2d(2)

        self.bottleneck_block1 = double_conv(512,512, time_emb_dims_exp)
        self.bottleneck_block2 = double_conv(512,512, time_emb_dims_exp)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256, time_emb_dims_exp)
        self.dconv_up2 = double_conv(128 + 256, 128, time_emb_dims_exp)
        self.dconv_up1 = double_conv(64 + 128, 64, time_emb_dims_exp)
        
        self.conv_last = ConvCLTLayerDet(64, channels, 1, isoutput=True)
        
        
    def forward(self, x, ts):
        time_emb = self.time_embeddings(ts)
        mu1 = self.dconv_down1(x, time_emb)
        x = self.maxpool(mu1)

        mu2 = self.dconv_down2(x, time_emb)
        x = self.maxpool(mu2)
        
        mu3 = self.dconv_down3(x, time_emb)
        x = self.maxpool(mu3)   
        
        x = self.dconv_down4(x, time_emb)
        
        y = torch.ones_like(x)*1e-6
        x, y = self.bottleneck_block1(x, y, time_emb)
        x, y = self.bottleneck_block2(x, y, time_emb)

        x = self.upsample(x)        
        y = self.upsample(y)        
        x = torch.cat([x, mu3], dim=1)
        y = torch.cat([y, torch.ones_like(mu3)*1e-6], dim=1)
        
        x,y = self.dconv_up3(x,y, time_emb)
        x = self.upsample(x)        
        y = self.upsample(y)        
        x = torch.cat([x, mu2], dim=1)
        y = torch.cat([y, torch.ones_like(mu2)*1e-6], dim=1)     


        x,y = self.dconv_up2(x,y, time_emb)
        x = self.upsample(x)        
        y = self.upsample(y)        
        x = torch.cat([x, mu1], dim=1)
        y = torch.cat([y, torch.ones_like(mu1)*1e-6], dim=1)    
        
        x,y = self.dconv_up1(x,y, time_emb)
        
        out, var = self.conv_last(x,y)
        
        return out, var
    


# Standard UNet structure
class UNetSimple(nn.Module):

    def __init__(self, n_class, base_channels, time_multiple = 2):
        super().__init__()
        time_emb_dims_exp = base_channels * time_multiple
        self.time_embeddings = SinusoidalPositionEmbeddings(time_emb_dims=base_channels, time_emb_dims_exp=time_emb_dims_exp)
        self.dconv_down1 = double_conv_no_clt(n_class, 64, time_emb_dims_exp)
        self.dconv_down2 = double_conv_no_clt(64, 128, time_emb_dims_exp)
        self.dconv_down3 = double_conv_no_clt(128, 256, time_emb_dims_exp)
        self.dconv_down4 = double_conv_no_clt(256, 512, time_emb_dims_exp)
        self.maxpool = nn.MaxPool2d(2)
        self.bottleneck_block1 = double_conv_no_clt(512,512, time_emb_dims_exp)
        self.bottleneck_block2 = double_conv_no_clt(512,512, time_emb_dims_exp)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dconv_up3 = double_conv_no_clt(256 + 512, 256, time_emb_dims_exp)
        self.dconv_up2 = double_conv_no_clt(128 + 256, 128, time_emb_dims_exp)
        self.dconv_up1 = double_conv_no_clt(128 + 64, 64, time_emb_dims_exp)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x,ts):
        time_emb = self.time_embeddings(ts)

        conv1 = self.dconv_down1(x, time_emb)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x, time_emb)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x, time_emb)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x, time_emb)
        
        x = self.bottleneck_block1(x, time_emb)
        x = self.bottleneck_block2(x, time_emb)

        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x, time_emb)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x, time_emb)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x, time_emb)
        
        out = self.conv_last(x)
        
        return out
