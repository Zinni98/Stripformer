import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torchvision import transforms


class GOPRODataset(Dataset):
    def __init__(self, root, img_transforms=None, training=True, gamma_blur=True):
        """
        Parameters
        ----------
        root : str
            directory containing the GoPro dataset
        img_transforms : torch.nn.Sequential
            transforms to be applied to the image
        training : bool
            true if training set, false for test set
        gamma_blur : bool
            whether to include gamma_blur dir for training
        """
        super().__init__()
        self.training = training
        if self.training:
            self.root = os.path.join(root, "train")
        else:
            self.root = os.path.join(root, "test")

        if img_transforms is None:
            # Default
            self.transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.CenterCrop((512, 512))])
        else:
            self.transforms = img_transforms
        self.gamma_blur = gamma_blur

        self.ground_truth_images, self.blur_images = self._get_images()
        self.length = len(self.ground_truth_images)
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def _get_images(self):
        ground_truth_images = []
        blur_images = []
        for filename in os.listdir(self.root):
            f = os.path.join(self.root, filename)
            if os.path.isdir(f):
                ground_truth_dir = os.path.join(f, "sharp")
                gt = self._get_imgs_from_path(ground_truth_dir)
                ground_truth_images.extend(gt)

                blur_dir = os.path.join(f, "blur")
                blur = self._get_imgs_from_path(blur_dir)
                blur_images.extend(blur)

                if self.gamma_blur:
                    blur_gamma_dir = os.path.join(f, "blur_gamma")
                    blur_gamma = self._get_imgs_from_path(blur_gamma_dir)
                    blur_images.extend(blur_gamma)
                    # Double because we are considering blur and blur gamma.
                    ground_truth_images.extend(gt)

        return ground_truth_images, blur_images

    def _get_imgs_from_path(self, path):
        res = []
        filenames = [filename.split(".")[0] for filename in os.listdir(path)]
        # Sorting to ensure correspondence between successive calls of the function
        filenames.sort()
        for filename in filenames:
            full_filename = filename + ".png"
            img_path = os.path.join(path, full_filename)
            img = Image.open(img_path)
            res.append(img)

        return res

    def __getitem__(self, index):
        blur_img, gt_img = self.transforms(self.blur_images[index],
                                           self.ground_truth_images[index])
        return blur_img, gt_img

    def __len__(self):
        return self.length


def get_data(root, batch_size, transforms):
    train_set = GOPRODataset(root,
                             img_transforms=transforms["train"])
    test_set = GOPRODataset(root,
                            training=False,
                            img_transforms=transforms["test"])

    train_loader = DataLoader(train_set,
                              batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_set,
                             batch_size,
                             shuffle=True)

    return train_set, test_set, train_loader, test_loader


def get_data_pretrain(root, batch_size, transforms):
    train_set, test_set, train_loader, test_loader = get_data(root,
                                                              batch_size,
                                                              transforms)

    pretrain_set = GOPRODataset(root,
                                img_transforms=transforms["pretrain"])
    pretrain_loader = DataLoader(pretrain_set,
                                 batch_size,
                                 shuffle=True)

    return (train_set,
            pretrain_set,
            test_set,
            train_loader,
            pretrain_loader,
            test_loader)


if __name__ == "__main__":
    ds = GOPRODataset("./GOPRO_Large")
    print(ds[0])
