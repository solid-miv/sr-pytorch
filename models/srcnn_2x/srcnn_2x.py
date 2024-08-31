import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
from PIL import Image


class SRCNN_2X(nn.Module):
    def __init__(self):
        super(SRCNN_2X, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)

    def forward(self, x):
        x = self.upsample(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    

def predict_srcnn_2x(model, image_path, device):
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        ToTensor(),
    ])

    low_res_img = Image.open(image_path).convert('RGB')

    img_tensor = transform(low_res_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    super_res_img = transforms.ToPILImage()(output.squeeze().cpu())

    low_res_shape = low_res_img.size
    super_res_shape = super_res_img.size

    return low_res_img, super_res_img, low_res_shape, super_res_shape


def load_srcnn_2x(model_path, device):
    model = SRCNN_2X()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")

    return model