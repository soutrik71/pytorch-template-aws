import lightning as L
import torch
from torch import nn, optim
from torchmetrics import Accuracy, Precision, Recall, F1Score
from timm.models import VisionTransformer


class ViTTinyClassifier(L.LightningModule):
    def __init__(
        self,
        img_size: int = 224,
        num_classes: int = 2,
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

        # Create ViT model
        self.model = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=False,
            pre_norm=pre_norm,
            global_pool="token",
        )

        # Metrics for multi-class classification
        metrics = {
            "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
            "precision": Precision(
                task="multiclass", num_classes=num_classes, average="macro"
            ),
            "recall": Recall(
                task="multiclass", num_classes=num_classes, average="macro"
            ),
            "f1": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
        }

        # Initialize metrics for each stage
        self.train_metrics = nn.ModuleDict(
            {name: metric.clone() for name, metric in metrics.items()}
        )
        self.val_metrics = nn.ModuleDict(
            {name: metric.clone() for name, metric in metrics.items()}
        )
        self.test_metrics = nn.ModuleDict(
            {name: metric.clone() for name, metric in metrics.items()}
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)

        # Get appropriate metric dictionary based on stage
        metrics = getattr(self, f"{stage}_metrics")
        metric_logs = {
            f"{stage}_{name}": metric(preds, y) for name, metric in metrics.items()
        }

        # Log metrics
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log_dict(metric_logs, prog_bar=True, on_step=False, on_epoch=True)
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
