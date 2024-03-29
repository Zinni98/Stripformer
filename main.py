import os
import sys
import config
from  transforms import *
from data import get_data, get_data_pretrain
from train import Trainer, TrainerPretrainer
from loss import StipformerLoss
from model.stripformer import Stripformer


def check_valid_dir(dir: str):
    """
    Check whether the path is either absolute or relative
    """
    return dir.startswith("./") or dir.startswith("/")


def get_dirs():
    incolab = "google.colab" in sys.modules

    if not (check_valid_dir(config.gopro_dir) and
            check_valid_dir(config.save_models_dir)):
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

    if config.save_models_dir.endswith(".tar"):
        if config.save_models_dir.startswith("./"):
            path_to_save_models = os.path.join(root_dir, config.save_models_dir[2:])
        else:
            path_to_save_models = config.save_models_dir
    else:
        raise ValueError("Invalid path, the filename should end with .tar extension")

    if config.load_dir:
        if config.load_dir.endswith(".tar"):
            if config.load_dir.startswith("./"):
                path_to_load_models = os.path.join(root_dir, config.load_dir[2:])
            else:
                path_to_load_models = config.load_dir
        else:
            raise ValueError("Invalid path, the filename \
                             should end with .tar extension")

    else:
        path_to_load_models = None

    return path_to_gopro, path_to_save_models, path_to_load_models


def main_fn():
    path_to_gopro, path_to_save_models, path_to_load_models = get_dirs()
    model = Stripformer()
    loss_fn = StipformerLoss()
    img_transforms = dict()
    img_transforms["train"] = Compose2Imgs([RandomHorizontalFlip2Imgs(),
                                            RandomVerticalFlip2Imgs(),
                                            RC2Imgs((config.train_img_size,  # noqa
                                                     config.train_img_size)),
                                            ToTensor2Imgs()])
    img_transforms["test"] = Compose2Imgs([ToTensor2Imgs()])
    if config.pretrain:
        img_transforms["pretrain"] = Compose2Imgs([RandomHorizontalFlip2Imgs(),
                                                   RandomVerticalFlip2Imgs(),
                                                   RC2Imgs((config.train_img_size,  # noqa
                                                            config.train_img_size)),
                                                    ToTensor2Imgs()])

        _, _, _, train_loader, pretrain_loader, test_loader = get_data_pretrain(path_to_gopro,  # noqa
                                                                                config.batch_size,  # noqa
                                                                                img_transforms  # noqa
                                                                                )
        trainer = TrainerPretrainer(config.epochs,
                                    config.pre_train_epochs,
                                    model,
                                    config.batch_size,
                                    loss_fn,
                                    pretrain_loader,
                                    train_loader,
                                    test_loader,
                                    path_to_save_models,
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
                          path_to_save_models,
                          path_to_load_models,
                          config.max_lr,
                          config.min_lr,
                          config.use_wandb,
                          config.accumulation_steps
                          )
    if config.test_only:
        loss, psnr, ssim = trainer.test_step()
        print(f"Test loss: {loss} \t Test pnsr: {psnr} \t\
                Test ssim: {ssim}\n")
    else:
        trainer.train()
        loss, psnr, ssim = trainer.test_step()
        print(f"Test loss: {loss} \t Test pnsr: {psnr} \t\
                Test ssim: {ssim}\n")


if __name__ == "__main__":
    main_fn()
