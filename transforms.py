from torchvision import transforms
import torchvision.transforms.functional as tvf
import torch
from torchvision.transforms.functional import InterpolationMode


class Compose2Imgs(transforms.Compose):
    def __init__(self, transforms):
        super().__init__(transforms)
    
    def __call__(self, img1, img2):
        for t in self.transforms:
            img1, img2 = t(img1, img2)
        return img1, img2


class ToTensor2Imgs(transforms.ToTensor):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, img1, img2):
        return tvf.to_tensor(img1), tvf.to_tensor(img2)

class CenterCrop2Imgs(transforms.CenterCrop):
    def __init__(self, size):
        super().__init__(size)
    
    def forward(self, img1, img2):
        return tvf.center_crop(img1, self.size), tvf.center_crop(img2, self.size)


class RC2Imgs(transforms.RandomCrop):
    def __init__(self, size):
        super().__init__(size, None, False, 0, "constant")

    def forward(self, img1, img2):
        if not img1.size == img1.size:
            raise ValueError("Images have two different sizes")
        
        i, j, h, w = self.get_params(img1, self.size)

        return tvf.crop(img1, i, j, h, w), tvf.crop(img2, i, j, h, w)
    
class RandomHorizontalFlip2Imgs(transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)
    
    def forward(self, img1, img2):
        if torch.rand(1) < self.p:
            img1 = tvf.hflip(img1)
            img2 = tvf.hflip(img2)
        return img1, img2
    
class RandomVerticalFlip2Imgs(transforms.RandomVerticalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)
    
    def forward(self, img1, img2):
        if torch.rand(1) < self.p:
            img1 = tvf.vflip(img1)
            img2 = tvf.vflip(img2)
        return img1, img2
