import lightning as L
import torch.nn.functional as F
from torch import optim
from torchmetrics.classification import Accuracy, F1Score
from timm.models import VisionTransformer
import torch


class ViTTinyClassifier(L.LightningModule):
    def __init__(
        self,
        img_size: int = 224,
        num_classes: int = 2,  # Binary classification with two classes
        embed_dim: int = 64,
        depth: int = 6,
        num_heads: int = 2,
        patch_size: int = 16,
        mlp_ratio: float = 3.0,
        pre_norm: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Vision Transformer model initialization
        self.model = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            pre_norm=pre_norm,
            global_pool="token",
        )

        # Define accuracy and F1 metrics for binary classification
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")

        self.train_f1 = F1Score(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.test_f1 = F1Score(task="binary")

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)  # Model output shape: [batch_size, num_classes]
        loss = F.cross_entropy(logits, y)  # Cross-entropy for binary classification
        preds = torch.argmax(logits, dim=1)  # Predicted class (0 or 1)

        # Update and log metrics
        acc = getattr(self, f"{stage}_acc")
        f1 = getattr(self, f"{stage}_f1")
        acc(preds, y)
        f1(preds, y)

        # Logging of metrics and loss
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_f1", f1, prog_bar=True, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.hparams.factor,
            patience=self.hparams.patience,
            min_lr=self.hparams.min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }


if __name__ == "__main__":
    model = ViTTinyClassifier()
    print(model)
