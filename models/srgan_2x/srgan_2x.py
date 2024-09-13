import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Constants
N_RESBLK_G = 20
UPSCALE = 2

# Generator model for SRGAN
class Generator(nn.Module):
    def __init__(self, n_res_blks, upscale_factor=4):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        self.prelu1 = nn.PReLU()
        self.res_blocks = nn.Sequential()
        for i in range(n_res_blks):
            self.res_blocks.add_module(f"res_blk_{i}",
                                       Residual_Block(in_channels=64, out_channels=64, strides=1, use_1x1_conv=False))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.pixel_shufflers = nn.Sequential()
        for i in range(int(np.log2(upscale_factor))):
            self.pixel_shufflers.add_module(f"pixel_shuffle_blk_{i}",
                                            PixelShufflerBlock(in_channels=64, upscale_factor=2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, padding=4)

    def forward(self, X):
        X = self.prelu1(self.conv1(X))
        X_before_resblks = X.clone()
        X = self.res_blocks(X)
        X = self.bn(self.conv2(X))
        X = F.relu(X + X_before_resblks)
        X = self.pixel_shufflers(X)
        X = self.conv3(X)

        return F.tanh(X)

# Residual block for SRGAN
class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, strides, use_1x1_conv=True):
        super(Residual_Block, self).__init__()
        self.use_1x1_conv = use_1x1_conv
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
        self.blk = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, X):
        X_original = X.clone()
        X = self.blk(X)
        if self.use_1x1_conv:
            X_original = self.conv1x1(X_original)

        return F.relu(X + X_original)
    
# Pixel shuffler block for SRGAN
class PixelShufflerBlock(nn.Module):
    def __init__(self, in_channels, upscale_factor=2):
        super(PixelShufflerBlock, self).__init__()
        self.blk = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=upscale_factor),
            nn.PReLU()
        )

    def forward(self, X):
        return self.blk(X)


def upscale_srgan_2x(input_image_path, model_path):
    """
    Upscales a low-resolution image by a factor of 2 using a pre-trained SRGAN model.

    Params:
        input_image_path (str): Path to the input low-resolution image.
        model_path (str): Path to the pre-trained SRGAN model checkpoint.

    Returns:
        tuple: A tuple containing the low-resolution image (PIL.Image) and the super-resolution image (PIL.Image).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load the trained generator model
    G = Generator(n_res_blks=N_RESBLK_G, upscale_factor=UPSCALE).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    G.load_state_dict(checkpoint['state_dict'])
    G.eval()
    
    low_res_img = Image.open(input_image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    input_tensor = transform(low_res_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        sr_tensor = G(input_tensor)
    
    super_res_image = transforms.ToPILImage()(sr_tensor.squeeze(0).cpu())

    return low_res_img, super_res_image