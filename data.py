import os
from torch.utils.data import Dataset
from PIL import Image
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
            whether to use gamma_blur dir or blur dir for blurred images
        """
        super().__init__()
        self.training = training
        if self.training:
            self.root = os.path.join(root, "train")
        else:
            self.root = os.path.join(root, "test")

        self.transforms = img_transforms
        self.gamma_blur = gamma_blur

        self.ground_truth_images, self.blur_images = self._get_images()

    def _get_images(self):
        ground_truth_images = []
        blur_images = []
        for filename in os.listdir(self.root):
            f = os.path.join(self.root, filename)
            if os.path.isdir(f):
                ground_truth_dir = os.path.join(f, "sharp")
                if self.gamma_blur:
                    blur_dir = os.path.join(f, "blur_gamma")
                else:
                    blur_dir = os.path.join(f, "blur")

                ground_truth_images.extend(self._get_imgs_from_path(ground_truth_dir))
                blur_images.extend(self._get_imgs_from_path(blur_dir))

        return ground_truth_images, blur_images

    def _get_imgs_from_path(self, path):
        res = []
        filenames = [filename.split(".")[0] for filename in os.listdir(path)]
        # Sorting to ensure correspondence between successive calls of the function
        filenames.sort()
        convert_to_tensor = transforms.ToTensor()
        for filename in filenames:
            full_filename = filename + ".png"
            img_path = os.path.join(path, full_filename)
            img = Image.open(img_path)
            tensor_img = convert_to_tensor(img)
            res.append(tensor_img)

        return res

    def __getitem__(self, index):
        return self.blur_images[index], self.ground_truth_images[index]

    def __len__(self):
        return len(self.ground_truth_images)


if __name__ == "__main__":
    ds = GOPRODataset("./GOPRO_Large")
    print(ds[0])
