from Baysian_Layers import ConvCLTLayerDet
import torch
from torchvision import transforms
from PIL import Image
from Models.UNetClosed import UNetNSDE

# Arrange
layer = ConvCLTLayerDet(1,1,3)
unet = UNetNSDE(1)
img = Image.open("5.png").convert("L")
convert_tensor = transforms.ToTensor()
img = convert_tensor(img)
img = img.view(1,*img.shape)
var = torch.ones_like(img)*1e-5

# Act
layerOuputMu, layerOutputVar = layer(img, var)
unetOuputMu, unetOutputVar = unet(img, var)

# Results
with torch.no_grad():
    print(torch.isnan(layerOuputMu).any())
    print(torch.isnan(layerOutputVar).any())
    print(torch.isnan(unetOuputMu).any())
    print(torch.isnan(unetOutputVar).any())


### Test 2 ###

# Arrange
var = torch.ones_like(img)*1e-5

# Act
layerOuputMu, layerOutputVar = layer(img, var)
unetOuputMu, unetOutputVar = unet(img, var)

# Results
with torch.no_grad():
    print(torch.isnan(layerOuputMu).any())
    print(torch.isnan(layerOutputVar).any())
    print(torch.isnan(unetOuputMu).any())
    print(torch.isnan(unetOutputVar).any())