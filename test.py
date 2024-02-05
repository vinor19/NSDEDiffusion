from Baysian_Layers import ConvCLTLayerDet
import torch
from torchvision import transforms
from PIL import Image
from Models.UNetClosed import UNetNSDE, PositionalEncoding, UNetSimple
from Models.UNetATT import SinusoidalPositionEmbeddings, UNetATT



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

class ModelConfig:
    BASE_CH = 16  # 64, 128, 256, 256
    BASE_CH_MULT = (1, 2, 4, 8) # 32, 16, 8, 4 
    APPLY_ATTENTION = (False, False, True, False)
    DROPOUT_RATE = 0.1
    TIME_EMB_MULT = 2 # 128
    DEVICE = "cpu"

# model = UNetATT(
#     input_channels          = img.shape[1],
#     output_channels         = img.shape[1],
#     base_channels           = ModelConfig.BASE_CH,
#     base_channels_multiples = ModelConfig.BASE_CH_MULT,
#     apply_attention         = ModelConfig.APPLY_ATTENTION,
#     dropout_rate            = ModelConfig.DROPOUT_RATE,
#     time_multiple           = ModelConfig.TIME_EMB_MULT,
# )
model = UNetSimple(1)
max = 0
print(img.shape[-1])
for i in range(0,100000):  
    ts = torch.ones(1, dtype=torch.long, device="cpu") * i
    # print(model(img,ts))
    eps = torch.randn_like(img).min()  # Noise
    if max > eps:
        max = eps
print(max)
