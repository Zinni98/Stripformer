import os
import sys
import config
from data import get_data
from train import Trainer
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

    if config.saved_models_dir.startswith("./"):
        path_to_saved_models = os.path.join(root_dir, config.saved_models_dir)
    else:
        path_to_saved_models = config.saved_models_dir

    return path_to_gopro, path_to_saved_models


def main_fn():
    path_to_gopro, path_to_saved_models = get_dirs()
    model = Stripformer()
    loss_fn = StipformerLoss()
    _, _, train_loader, test_loader = get_data(path_to_gopro,
                                               config.batch_size)
    trainer = Trainer(config.epochs,
                      model,
                      config.batch_size,
                      loss_fn,
                      train_loader,
                      test_loader,
                      path_to_saved_models,
                      config.max_lr,
                      config.min_lr,
                      config.wandb
                      )
    trainer.train()
