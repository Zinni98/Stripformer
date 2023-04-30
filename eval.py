import torch
import utils
import matplotlib.pyplot as plt
from einops import rearrange
from torchvision.utils import save_image



def save_img(tensor_img, path ="example_images/img1.png"):
    if len(tensor_img.shape) == 4:
        tensor_img = torch.unsqueeze(tensor_img, 0)
    save_image(tensor_img, path)
    tensor_img = rearrange(tensor_img, "c h w -> h w c")
    plt.imshow(tensor_img)

def deblur_img(model, blurred_img):
    if len(tensor_img.shape) == 3:
        tensor_img = torch.squeeze(tensor_img, 0)
    
    deblurred_img = model(blurred_img)




