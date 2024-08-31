import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
from PIL import Image


class SRCNN_4X(nn.Module):
    def __init__(self):
        super(SRCNN_4X, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)

    def forward(self, x):
        x = self.upsample(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        return x
    

def predict_srcnn_4x(model, image_path, device):
    """
    Predicts the super-resolution image using the SRCNN 4x model.

    Params:
        model (torch.nn.Module): The trained SRCNN 4x model.
        image_path (str): The path to the low-resolution image.
        device (torch.device): The device to perform the prediction on.

    Returns:
        tuple: A tuple containing the following elements:
            - low_res_img (PIL.Image.Image): The low-resolution input image.
            - super_res_img (PIL.Image.Image): The super-resolution output image.
            - low_res_shape (tuple): The shape of the low-resolution image (width, height).
            - super_res_shape (tuple): The shape of the super-resolution image (width, height).
    """
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        ToTensor(),
    ])

    low_res_img = Image.open(image_path).convert('RGB')

    # convert to tensor and add batch dimension
    img_tensor = transform(low_res_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    # convert output tensor to image
    super_res_img = transforms.ToPILImage()(output.squeeze().cpu())

    # get shapes (debugging)
    low_res_shape = low_res_img.size
    super_res_shape = super_res_img.size

    return low_res_img, super_res_img, low_res_shape, super_res_shape


def load_srcnn_4x(model_path, device):
    """
    Loads the SRCNN 4x model from the specified model_path and moves it to the specified device.

    Params:
        model_path (str): The path to the saved model file.
        device (torch.device): The device to move the model to.

    Returns:
        torch.nn.Module: The loaded SRCNN 4x model.
    """
    model = SRCNN_4X()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")

    return model