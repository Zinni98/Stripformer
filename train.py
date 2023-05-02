import torch
import wandb
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts


class Trainer(nn.Module):
    def __init__(self,
                 epochs: int,
                 network: nn.Module,
                 batch_size: int,
                 loss_fn,
                 go_pro_train_loader,
                 go_pro_test_loader=None,
                 save_path: str = None,
                 load_from_path: str = None,
                 max_lr: float = 1e-4,
                 min_lr: float = 1e-7,
                 use_wandb: bool = False,
                 accumulation_steps: int = 0):
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
        self.current_epoch = 0
        self.network = network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = go_pro_train_loader
        self.test_loader = go_pro_test_loader
        self.lr = max_lr
        self.min_lr = min_lr
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.save_path = save_path
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure().to(self.device)
        self.accumulation_steps = accumulation_steps

        self.load_from_path = load_from_path

        if self.load_from_path:
            self.checkpoint = torch.load(self.load_from_path)
            self.network.load_state_dict(self.checkpoint["model_state_dict"])
            self.network.to(self.device)
            self.optimizer = Adam(self.network.parameters(), self.min_lr, amsgrad=True)
            self.scheduler = CosineAnnealingLR(self.optimizer,
                                               int(self.epochs/25),
                                               eta_min=self.lr)
            self.optimizer.load_state_dict(self.checkpoint["optim_state_dict"])
            self.scheduler.load_state_dict(self.checkpoint["sched_state_dict"])
            self.current_epoch = self.checkpoint["epoch"]
        else:
            self.network.to(self.device)
            self.optimizer = Adam(self.network.parameters(), self.min_lr, amsgrad=True)
            self.scheduler = CosineAnnealingLR(self.optimizer,
                                               int(self.epochs/25),
                                               eta_min=self.lr)

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

    def save_state_dict(self, e):
        torch.save({"model_state_dict": self.network.state_dict(),
                    "optim_state_dict": self.optimizer.state_dict(),
                    "sched_state_dict": self.scheduler.state_dict(),
                    "epoch": e
                    }, self.save_path)

    def train(self, loader=None):
        loader = loader if loader else self.train_loader
        best = 0
        for e in range(self.current_epoch, self.epochs):
            print(f"----------- Epoch {e+1} -----------")
            self.network.train()
            train_loss, train_psnr, train_ssim = self._training_epoch(loader)
            print(f"\nTraining loss: {train_loss} \t Training pnsr: {train_psnr} \t\
                    Training ssim: {train_ssim}\n")

            if self.save_path and train_psnr > best:
                best = train_psnr
                self.save_state_dict(e)

        if self.wandb:
            wandb.finish()

    def _training_epoch(self, loader):
        samples = 0
        cumulative_loss = 0
        cumulative_psnr = 0
        cumulative_ssim = 0
        loader = loader
        with tqdm(loader, unit="batch") as tepoch:
            for batch_idx, imgs in enumerate(tepoch):
                blur_img = imgs[0]
                sharp_img = imgs[1]
                self.optimizer.zero_grad()
                tepoch.set_description(f"{batch_idx} Batch")

                blur_img = blur_img.to(self.device)
                sharp_img = sharp_img.to(self.device)

                with torch.cuda.amp.autocast():
                    out = self.network(blur_img)
                    if self.accumulation_steps == 0:
                        loss = self.loss_fn(out, sharp_img, blur_img)
                    else:
                        # Mean reduction of the loss
                        loss = self.loss_fn(out, sharp_img, blur_img) / self.accumulation_steps # noqa

                self._scaler.scale(loss).backward()

                if (((batch_idx + 1) % self.accumulation_steps == 0) or
                   (batch_idx + 1) == len(loader)):
                    self._scaler.step(self.optimizer)
                    # self.optimizer.step()
                    self._scaler.update()

                samples += blur_img.shape[0]
                cumulative_loss += loss.item()

                with torch.no_grad():
                    psnr = self.psnr(out, sharp_img)
                    ssim = self.ssim(out, sharp_img)
                    cumulative_psnr += psnr.item()
                    cumulative_ssim += ssim.item()

                tepoch.set_postfix({"psnr": cumulative_psnr/samples,
                                    "ssim": cumulative_ssim/samples,
                                    "loss": cumulative_loss/samples,})

        if self.wandb:
            wandb.log({"psnr": cumulative_psnr/samples,
                       "ssim": cumulative_ssim/samples,
                       "loss": cumulative_loss/samples,
                       "lr": self._get_lr()})
        self.scheduler.step()
        return cumulative_loss/samples, cumulative_psnr/samples, cumulative_ssim/samples

    def _get_lr(self):
        for pg in self.optimizer.param_groups:
            return pg["lr"]

    def _test_step(self):
        samples = 0
        cumulative_loss = 0
        cumulative_psnr = 0
        cumulative_ssim = 0
        self.network.eval()
        with torch.no_grad():
            with tqdm(self.test_loader, unit="batch") as tepoch:
                for batch_idx, imgs in enumerate(tepoch):
                    tepoch.set_description(f"{batch_idx} Batch")

                    blur_img = imgs[0].to(self.device)
                    sharp_img = imgs[1].to(self.device)

                    with torch.cuda.amp.autocast():
                        out = self.network(blur_img)
                        loss = self.loss_fn(out, sharp_img, blur_img)

                    samples += blur_img.shape[0]
                    cumulative_loss += loss.item()

                    psnr = self.psnr(out, sharp_img)
                    ssim = self.ssim(out, sharp_img)
                    cumulative_psnr += psnr.item()
                    cumulative_ssim += ssim.item()
                    tepoch.set_postfix({"psnr": cumulative_psnr/samples,
                                        "ssim": cumulative_ssim/samples,
                                        "loss": cumulative_loss/samples},)

        return cumulative_loss/samples, cumulative_psnr/samples, cumulative_ssim/samples


class TrainerPretrainer(Trainer):
    """
    Same as trainer, but handles pretrain, managing
    saving and restoring the model status considering
    if pretrain has finished or not.
    """
    def __init__(self,
                 train_epochs: int,
                 pre_train_epochs: int,
                 network: nn.Module,
                 batch_size: int,
                 loss_fn,
                 go_pro_pre_train_loader,
                 go_pro_train_loader,
                 go_pro_test_loader=None,
                 save_path: str = None,
                 load_from_path: str = None,
                 max_lr: float = 1e-4,
                 min_lr: float = 1e-7,
                 use_wandb: bool = False,
                 accumulation_steps: int = 0):

        super().__init__(0,
                         network,
                         batch_size,
                         loss_fn,
                         go_pro_train_loader,
                         go_pro_test_loader,
                         save_path,
                         load_from_path,
                         max_lr,
                         min_lr,
                         False,  # Set here wandb
                         accumulation_steps)
        self.pre_train_loader = go_pro_pre_train_loader
        if load_from_path:
            try:
                self.pretrain_done = self.checkpoint["pre_train_done"]
            except KeyError:
                self.pre_train_done = True
        else:
            self.pre_train_done = False

        self.pre_train_epochs = pre_train_epochs
        self.train_epochs = train_epochs

        self.epochs = train_epochs if self.pre_train_done else pre_train_epochs

        # Redefinition because I need to know the number of pre_train_epochs
        self.scheduler = CosineAnnealingLR(self.optimizer,
                                           self.epochs,
                                           eta_min=self.min_lr)
        self.wandb = use_wandb
        if self.wandb:
            self.run = wandb.init(project="stripformer",
                                  tags=["stripformer", "siv"])

            wandb.config = {
                "epochs": self.epochs,
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                "model": "stripformer"
            }

    def save_state_dict(self, e):
        torch.save({"model_state_dict": self.network.state_dict(),
                    "optim_state_dict": self.optimizer.state_dict(),
                    "sched_state_dict": self.scheduler.state_dict(),
                    "epoch": e,
                    "pre_train_done": self.pre_train_done
                    }, self.save_path)

    def train(self):
        if self.pre_train_done:
            super().train()
        else:
            super().train(self.pre_train_loader)
            self.pre_train_done = True
            # Epochs = 0 because training should be done
            self.save_state_dict(0)
            self.epochs = self.train_epochs
            # Redefinition with new number of epochs
            self.scheduler = CosineAnnealingLR(self.optimizer,
                                               self.epochs,
                                               eta_min=self.min_lr)
            super().train()
