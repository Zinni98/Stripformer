import config
from train import Trainer
from model.stripformer import Stripformer
from loss import StipformerLoss
from data import get_data


def main():
    model = Stripformer()
    loss_fn = StipformerLoss()
    _, _, train_loader, test_loader = get_data(config.path_to_gopro,
                                               config.batch_size)
    trainer = Trainer(config.epochs,
                      model,
                      config.batch_size,
                      loss_fn,
                      train_loader,
                      test_loader,
                      config.path_to_saved_models,
                      config.max_lr,
                      config.min_lr,
                      config.wandb
                      )
    trainer.train()


if __name__ == "__main__":
    main()
