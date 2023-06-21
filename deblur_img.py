import torch
import matplotlib.pyplot as plt
from einops import rearrange
from torchvision.utils import save_image
import torchvision.transforms as T
from model.stripformer import Stripformer
from PIL import Image


def save_img(tensor_img, path="./example_images/deblurred/img2.png"):
    if len(tensor_img.shape) == 4:
        tensor_img = torch.squeeze(tensor_img, 0)
    save_image(tensor_img, fp=path)
    tensor_img = rearrange(tensor_img, "c h w -> h w c")
    img = tensor_img.detach().numpy()
    plt.imshow(img)


def deblur_img(model, blurred_img):
    if not isinstance(blurred_img, torch.Tensor):
        to_tensor = T.ToTensor()
        blurred_img = to_tensor(blurred_img)
        blurred_img = blurred_img
    if len(blurred_img.shape) == 3:
        blurred_img = torch.unsqueeze(blurred_img, 0)
    deblurred_img = model(blurred_img)
    return deblurred_img


def main():
    network = Stripformer()
    checkpoint = torch.load("./saved_models/run4.tar", map_location="cpu")
    network.load_state_dict(checkpoint["model_state_dict"])
    network.to("cpu")
    print("network loaded")
    img = Image.open("./GOPRO_Large/test/GOPR0384_11_00/blur/000008.png").convert("RGB") # ./GOPRO_Large/test/GOPR0384_11_00/blur/000008.png
    db_img = deblur_img(network, img)
    save_img(db_img)

if __name__ == "__main__":
    main()