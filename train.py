import torch
import torch.nn as nn


class Trainer(nn.Module):
    def __init__(self, epochs, network):
        """
        Parameters
        ----------
        epochs : int
            Number of epochs for training
        
        network : torch.nn.Module

        """
        super().__init__()
        self.epochs = epochs
        self.network = network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        self.network.train()
        samples = 0
        cumulative_loss = 0

        for e in self.epochs:



