import torch
import wandb
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchmetrics import PeakSignalNoiseRatio
from torch.optim.lr_scheduler import CosineAnnealingLR


class Trainer(nn.Module):
    def __init__(self,
                 epochs: int,
                 network: nn.Module,
                 batch_size: int,
                 loss_fn,
                 go_pro_train_loader,
                 go_pro_test_loader=None,
                 save_path: str = None,
                 max_lr: float = 1e-4,
                 min_lr: float = 1e-7,
                 use_wandb: bool = False):
        """
        Parameters
        ----------
        epochs : int
            Number of epochs for training.

        network : torch.nn.Module
            Stripformer network.

        go_pro_train_loader : torch.utils.data.Dataloader
            Training dataloader for the go_pro dataset.

        go_pro_test_loader : torch.utils.data.Dataloader
            Test dataloader for the go_pro dataset. If given,
            it is used to see wether the model is not over-fitting
            during raining.

        save_path : str
            Path where to save the trained model. If None, the model won't be
            saved.
        """
        super().__init__()
        self.epochs = epochs
        self.network = network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = go_pro_train_loader
        self.test_loader = go_pro_test_loader
        self.lr = max_lr
        self.min_lr = min_lr
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.optimizer = Adam(self.network.parameters(), self.lr, amsgrad=True)
        self.scheduler = CosineAnnealingLR(self.optimizer,
                                           self.epochs,
                                           eta_min=self.min_lr)
        self.save_path = save_path
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.wandb = use_wandb
        self._scaler = torch.cuda.amp.GradScaler()

        if self.wandb:
            self.run = wandb.init(project="stripformer",
                                  tags=["stripformer", "siv"])

            wandb.config = {
                "epochs": self.epochs,
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                "model": "stripformer"
            }

    def train(self):
        self.network.train()
        self.network.to(self.device)

        for e in range(self.epochs):
            print(f"----------- Epoch {e+1} -----------")
            train_loss, train_psnr = self._training_step()
            test_loss, test_psnr = self._test_step()
            print(f"Training loss: {train_loss} \t Training pnsr: {train_psnr} \n")
            print(f"Test loss: {test_loss} \t Test pnsr: {test_psnr} \n")

        if self.save_path:
            torch.save(self.network.state_dict(), self.save_path)

        if self.wandb:
            wandb.finish()

    def _training_step(self):
        samples = 0
        cumulative_loss = 0
        cumulative_psnr = 0
        with tqdm(self.train_loader, unit="batch") as tepoch:
            for batch_idx, imgs in enumerate(tepoch):
                blur_img = imgs[0]
                sharp_img = imgs[1]
                self.optimizer.zero_grad()
                tepoch.set_description(f"{batch_idx} Batch")

                blur_img = blur_img.to(self.device)
                sharp_img = sharp_img.to(self.device)

                with torch.cuda.amp.autocast():
                    out = self.network(blur_img)
                    loss = self.loss_fn(out, sharp_img, blur_img)

                self._scaler.scale(loss).backward()

                self._scaler.step(self.optimizer)
                # self.optimizer.step()
                self._scaler.update()

                samples += blur_img.shape[0]
                cumulative_loss += loss.item()

                psnr = self.psnr(out, sharp_img)
                cumulative_psnr += psnr.item()

        if self.wandb:
            wandb.log({"psnr": cumulative_psnr/samples,
                       "loss": cumulative_loss/samples})
        self.scheduler.step()

    def _test_step(self):
        samples = 0
        cumulative_loss = 0
        cumulative_psnr = 0
        self.network.eval()
        with torch.no_grad:
            with tqdm(self.train_loader, unit="batch") as tepoch:
                for batch_idx, blur_img, sharp_img in enumerate(tepoch):
                    tepoch.set_description(f"{batch_idx} Batch")

                    blur_img = blur_img.to(self.device)
                    sharp_img = sharp_img.to(self.device)

                    with torch.cuda.amp.autocast():
                        out = self.network(blur_img)
                        loss = self.loss_fn(out, sharp_img, blur_img)

                    samples += blur_img.shape[0]
                    cumulative_loss += loss.item()

                    psnr = self.psnr(out, sharp_img)
                    cumulative_psnr += psnr.item()

        return cumulative_loss/samples, cumulative_psnr/samples
