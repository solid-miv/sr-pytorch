import torch
from torch import nn
from torchvision import transforms
from PIL import Image


class ResidualBlock(nn.Module):
    def __init__(self, n_features):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_features, n_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_features, n_features, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out *= 0.1
        out += residual

        return out


class EDSR_2X(nn.Module):
    def __init__(self, n_resblocks=16, n_features=64):
        super(EDSR_2X, self).__init__()

        # initial convolution
        self.conv_first = nn.Conv2d(3, n_features, kernel_size=3, padding=1)

        # residual blocks
        self.body = nn.Sequential(
            *[ResidualBlock(n_features) for _ in range(n_resblocks)]
        )

        # feature map conv
        self.conv_body = nn.Conv2d(n_features, n_features, kernel_size=3, padding=1)

        # upsampling (2x)
        self.upsampling = nn.Sequential(
            nn.Conv2d(n_features, n_features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )

        # final output layer
        self.conv_last = nn.Conv2d(n_features, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_first(x)
        residual = x
        x = self.body(x)
        x = self.conv_body(x)
        x *= 0.1
        x += residual
        x = self.upsampling(x)
        x = self.conv_last(x)

        return x
    

def predict_edsr_2x(model, image_path, device):
    model = model.to(device)
    model.eval()

    # transforms
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    lr_img = Image.open(image_path).convert('RGB')

    # convert to tensor and add batch dimension
    lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)

    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    sr_img = to_pil(sr_tensor.squeeze().cpu().clamp(0, 1))

    # shapes
    lr_shape = lr_img.size
    sr_shape = sr_img.size

    return lr_img, sr_img, lr_shape, sr_shape


def load_edsr_2x(model_path, device):
    model = EDSR_2X()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")

    return model