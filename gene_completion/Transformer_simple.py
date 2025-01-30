from Transformer_encoder_decoder import TransformerEncoder, TransformerDecoder
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
from utils import *
import random
import torch

class Transformer(pl.LightningModule):

    def __init__(self,
                 input_dim: int, 
                 latent_dim: int, 
                 output_dim: int,
                 embedding_dim: int,
                 num_layers: int,
                 num_heads: int,
                 lr=1e-5,
                 alpha=0.9,
                 gene_weights=None) -> None:
        super().__init__()
        
        
        self.Encoder = TransformerEncoder(input_dim=input_dim, 
                               embedding_dim=embedding_dim, 
                               latent_dim=latent_dim, 
                               num_heads=num_heads, 
                               num_layers=num_layers, 
                               dropout=0.1)
        
        self.Decoder = TransformerDecoder(input_dim=input_dim, 
                               embedding_dim=embedding_dim, 
                               latent_dim=latent_dim, 
                               num_heads=num_heads, 
                               num_layers=4, 
                               dropout=0.1)

        self.lr = lr
        self.weights = gene_weights
        self.alpha = alpha 

    def encoder(self, x):
        # (batch, seq_len, d_model)
        x = self.Encoder(x)
        return x
    
    def decoder(self, encoder_output):
        # (batch, seq_len, d_model)
        #breakpoint()
        x = self.Decoder(encoder_output)
        return x
        
    def forward(self, x, batch_size):
        """The forward function takes in an image and returns the reconstructed image."""
        encoder_output = self.encoder(x)
        #breakpoint()
        if self.training:
            if random.random() < 0.5:  # 50% chance to add noise
                encoder_output = add_noise(encoder_output, noise_factor=0.1)
            else:
                encoder_output = encoder_output
        #breakpoint()
        encoder_output = encoder_output[:,0,:]
        decoder_output = self.decoder(encoder_output)
        return decoder_output

    def _get_reconstruction_loss(self, x, x_hat, mask):
        """
        Computes the reconstruction loss with a focus on the central spot.

        Parameters
        ----------
        x : torch.Tensor
            Original input tensor of shape (batch_size, seq_len=7, input_size=1024).
        x_hat : torch.Tensor
            Reconstructed tensor of shape (batch_size, seq_len=7, input_size=1024).

        Returns
        ----------
        torch.Tensor
            Weighted reconstruction loss.
        """
        # Central spot reconstruction (first row in each sequence)
        #weights_test = self.weights.unsqueeze(0).repeat(x.size(0), x.size(1), 1)
        #weights_test = self.weights.unsqueeze(0).repeat(x.size(0),  1)
        #important_mask = (weights_test == 1).bool()
        #auxiliary_mask = (weights_test == 0).bool()
        #breakpoint()
        mask = mask[:,0,:]
        x = x[:,0,:]
        important_mask = (mask == 1).bool()
        auxiliary_mask = (mask == 0).bool()
        
        important_loss = F.mse_loss(x[important_mask], x_hat[important_mask])
        auxiliary_loss = F.mse_loss(x[auxiliary_mask], x_hat[auxiliary_mask])

        # Weighted total loss
        total_loss = self.alpha * important_loss + (1 - self.alpha) * auxiliary_loss 
        #total_loss = F.mse_loss(x[important_mask], x_hat.flatten())
        #total_loss = F.mse_loss(x[important_mask], x_hat[important_mask])
        
        if torch.isnan(total_loss):
            print("Loss is NaN! Inspect inputs, model outputs, or weights.")
            breakpoint()
            
        #self.log("important_loss", important_loss, prog_bar=True, logger=True)
        #self.log("auxiliary_loss", auxiliary_loss, prog_bar=True, logger=True)
        self.log("total_loss", total_loss, prog_bar=True, logger=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=10, min_lr=1e-10)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.1)
        #return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        return {"optimizer": optimizer, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        x = batch[0].float()
        # Randomly decide to add noise
        #if random.random() < 0.5:  # 50% chance to add noise
        #    noisy_x = add_noise(x, noise_factor=0.1)
        #else:
        #    noisy_x = x  # Use clean data
        noisy_x = x
        batch_size = x.shape[0]
        mask = batch[1]
        x_hat = self.forward(noisy_x, batch_size)
        loss = self._get_reconstruction_loss(x, x_hat, mask)
        
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", current_lr, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    """   
    def on_after_backward(self):
        # Aquí inspeccionas los gradientes después de calcular backward
        print("=== Inspecting Gradients ===")
        for name, param in self.named_parameters():
            if param.grad is not None:
                print(f"{name}: grad max={param.grad.abs().max().item()}, grad min={param.grad.abs().min().item()}")
    """
    def validation_step(self, batch, batch_idx):
        x = batch[0].float()
        batch_size = x.shape[0]
        mask = batch[1]
        x_hat = self.forward(x, batch_size)
        loss = self._get_reconstruction_loss(x, x_hat, mask)
        self.log("val_loss", loss, logger=True)

    def test_step(self, batch, batch_idx):
        x = batch[0].float()
        batch_size = x.shape[0]
        mask = batch[1]
        x_hat = self.forward(x, batch_size)
        loss = self._get_reconstruction_loss(x, x_hat, mask)
        self.log("test_loss", loss, logger=True)
