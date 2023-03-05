import os
import sys
import config
from data import get_data, get_data_pretrain
from train import Trainer, TrainerPretrainer
from loss import StipformerLoss
from model.stripformer import Stripformer
from torchvision import transforms


def check_valid_dir(dir: str):
    """
    Check whether the path is either absolute or relative
    """
    return dir.startswith("./") or dir.startswith("/")


def get_dirs():
    incolab = "google.colab" in sys.modules

    if not (check_valid_dir(config.gopro_dir) and
            check_valid_dir(config.saved_models_dir)):
        raise ValueError("Invalid path, check on config.py whether\
                          paths start with './' or '/'")

    if incolab:
        root_dir = config.colab_dir
    else:
        root_dir = os.path.dirname(os.path.realpath(__file__))

    if config.gopro_dir.startswith("./"):
        path_to_gopro = os.path.join(root_dir, config.gopro_dir[2:])
    else:
        path_to_gopro = config.gopro_dir

    if config.saved_models_dir.endswith(".tar"):
        if config.saved_models_dir.startswith("./"):
            path_to_saved_models = os.path.join(root_dir, config.saved_models_dir[2:])
        else:
            path_to_saved_models = config.saved_models_dir
    else:
        raise ValueError("Invalid path, the filename should end with .tar extention")

    if config.load_dir.endswith(".tar"):
        if config.load_dir.startswith("./"):
            path_to_load_models = os.path.join(root_dir, config.load_dir[2:])
        else:
            path_to_load_models = config.load_dir
    else:
        raise ValueError("Invalid path, the filename should end with .tar extension")

    return path_to_gopro, path_to_saved_models, path_to_load_models


def main_fn():
    path_to_gopro, path_to_saved_models, path_to_load_models = get_dirs()
    model = Stripformer()
    loss_fn = StipformerLoss()
    img_transforms = dict()
    img_transforms["train"] = transforms.Compose([transforms.ToTensor(),
                                                  transforms.CenterCrop((config.train_img_size,  # noqa
                                                                         config.train_img_size))])  # noqa
    img_transforms["test"] = transforms.Compose([transforms.ToTensor(),
                                                 transforms.CenterCrop((config.train_img_size,  # noqa
                                                                        config.train_img_size))])  # noqa
    if config.pretrain:
        img_transforms["pretrain"] = transforms.Compose([transforms.ToTensor(),
                                                         transforms.CenterCrop((config.pre_train_img_size,  # noqa
                                                                                config.pre_train_img_size))])  # noqa

        _, _, _, train_loader, pretrain_loader, test_loader = get_data_pretrain(path_to_gopro,  # noqa
                                                                                config.batch_size,  # noqa
                                                                                img_transforms  # noqa
                                                                                )
        trainer = TrainerPretrainer(config.epochs,
                                    model,
                                    config.batch_size,
                                    loss_fn,
                                    pretrain_loader,
                                    train_loader,
                                    test_loader,
                                    path_to_saved_models,
                                    path_to_load_models,
                                    config.max_lr,
                                    config.min_lr,
                                    config.use_wandb,
                                    config.accumulation_steps
                                    )
    else:

        _, _, train_loader, test_loader = get_data(path_to_gopro,
                                                   config.batch_size,
                                                   img_transforms
                                                   )

        trainer = Trainer(config.epochs,
                          model,
                          config.batch_size,
                          loss_fn,
                          train_loader,
                          test_loader,
                          path_to_saved_models,
                          path_to_load_models,
                          config.max_lr,
                          config.min_lr,
                          config.use_wandb,
                          config.accumulation_steps
                          )
    trainer.train()


if __name__ == "__main__":
    main_fn()
