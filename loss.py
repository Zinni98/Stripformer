import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3, reduction="mean"):
        super().__init__()
        if reduction != "mean" and reduction != "sum":
            raise ValueError("Reduction type not supported")
        else:
            self.reduction = reduction
        self.eps = eps

    def forward(self, output, target):
        diff = output - target

        out = torch.sqrt((diff * diff) + (self.eps * self.eps))
        if self.reduction == "mean":
            out = torch.mean(out)
        else:
            out = torch.sum(out)

        return out


class EdgeLoss(nn.Module):
    def __init__(self):
        """
        Taken from:
        https://github.com/swz30/MPRNet/blob/main/Deblurring/losses.py
        """
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down*4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(weights='VGG19_Weights.DEFAULT').features # noqa
        self.slice1 = torch.nn.Sequential()

        for x in range(12):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        return h_relu1


class ContrastLoss(nn.Module):
    """
    Taken from:
    https://github.com/pp00704831/Stripformer/blob/main/models/losses.py
    """
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg = Vgg19().to(device)
        self.l1 = nn.L1Loss()
        self.ab = ablation
        self.down_sample_4 = nn.Upsample(scale_factor=1 / 4, mode='bilinear')

    def forward(self, restore, sharp, blur):
        B, C, H, W = restore.size()
        restore_vgg, sharp_vgg, blur_vgg = self.vgg(restore), self.vgg(sharp), self.vgg(blur)  # noqa: ignore

        # filter out sharp regions
        threshold = 0.01
        mask = torch.mean(torch.abs(sharp-blur), dim=1).view(B, 1, H, W)
        mask[mask <= threshold] = 0
        mask[mask > threshold] = 1
        mask = self.down_sample_4(mask)
        d_ap = torch.mean(torch.abs((restore_vgg - sharp_vgg.detach())),
                          dim=1).view(B, 1, H//4, W//4)
        d_an = torch.mean(torch.abs((restore_vgg - blur_vgg.detach())),
                          dim=1).view(B, 1, H//4, W//4)
        mask_size = torch.sum(mask)
        contrastive = torch.sum((d_ap / (d_an + 1e-7)) * mask) / mask_size

        return contrastive


class StipformerLoss(nn.Module):
    def __init__(self, lambda1=0.05, lambda2=0.0005):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.char = CharbonnierLoss()
        self.edge = EdgeLoss()
        self.contrastive = ContrastLoss()

    def forward(self, restore, sharp, blur):
        char = self.char(restore, sharp)
        edge = self.lambda1 * self.edge(restore, sharp)
        contr = self.lambda2 * self.contrastive(restore, sharp, blur)
        loss = char + edge + contr
        return loss
