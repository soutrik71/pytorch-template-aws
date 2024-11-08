import os
import shutil
from pathlib import Path
from typing import List
import torch
import lightning as L
from lightning.pytorch.loggers import Logger, TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    RichModelSummary,
    RichProgressBar,
)
from dotenv import load_dotenv, find_dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
from src.datamodules.catdog_datamodule import CatDogImageDataModule
from src.utils.logging_utils import setup_logger, task_wrapper
from loguru import logger
import rootutils

# Load environment variables
load_dotenv(find_dotenv(".env"))

# Setup root directory

root = rootutils.setup_root(__file__, indicator=".project-root")


def initialize_callbacks(cfg: DictConfig) -> List[L.Callback]:
    """Initialize callbacks based on configuration."""
    callback_classes = {
        "model_checkpoint": ModelCheckpoint,
        "early_stopping": EarlyStopping,
        "rich_model_summary": RichModelSummary,
        "rich_progress_bar": RichProgressBar,
    }
    return [callback_classes[name](**params) for name, params in cfg.callbacks.items()]


def initialize_loggers(cfg: DictConfig) -> List[Logger]:
    """Initialize loggers based on configuration."""
    logger_classes = {
        "tensorboard": TensorBoardLogger,
        "csv": CSVLogger,
    }
    return [logger_classes[name](**params) for name, params in cfg.logger.items()]


def load_checkpoint_if_available(ckpt_path: str) -> str:
    """Return the checkpoint path if available, else None."""
    if ckpt_path and Path(ckpt_path).exists():
        logger.info(f"Using checkpoint: {ckpt_path}")
        return ckpt_path
    logger.warning(f"Checkpoint not found at {ckpt_path}. Using current model weights.")
    return None


def clear_checkpoint_directory(ckpt_dir: str):
    """Clear checkpoint directory contents without removing the directory."""
    ckpt_dir_path = Path(ckpt_dir)
    if not ckpt_dir_path.exists():
        logger.info(f"Creating checkpoint directory: {ckpt_dir}")
        ckpt_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        logger.info(f"Clearing checkpoint directory: {ckpt_dir}")
        for item in ckpt_dir_path.iterdir():
            try:
                item.unlink() if item.is_file() else shutil.rmtree(item)
            except Exception as e:
                logger.error(f"Failed to delete {item}: {e}")


@task_wrapper
def train_module(
    data_module: L.LightningDataModule, model: L.LightningModule, trainer: L.Trainer
):
    """Train the model and log metrics."""
    logger.info("Starting training")
    trainer.fit(model, data_module)
    train_metrics = trainer.callback_metrics
    train_acc = train_metrics.get("train_acc")
    val_acc = train_metrics.get("val_acc")
    logger.info(
        f"Training completed. Metrics - train_acc: {train_acc}, val_acc: {val_acc}"
    )
    return train_metrics


@task_wrapper
def run_test_module(
    cfg: DictConfig,
    datamodule: L.LightningDataModule,
    model: L.LightningModule,
    trainer: L.Trainer,
):
    """Test the model using the best checkpoint or current model weights."""
    logger.info("Starting testing")
    datamodule.setup(stage="test")
    test_metrics = trainer.test(
        model, datamodule, ckpt_path=load_checkpoint_if_available(cfg.ckpt_path)
    )
    logger.info(f"Test metrics: {test_metrics}")
    return test_metrics[0] if test_metrics else {}


@hydra.main(config_path="../configs", config_name="train", version_base="1.1")
def setup_run_trainer(cfg: DictConfig):
    """Set up and run the Trainer for training and testing."""
    # Display configuration
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize logger
    log_path = Path(cfg.paths.log_dir) / (
        "train.log" if cfg.task_name == "train" else "eval.log"
    )
    setup_logger(log_path)

    # Display key paths
    for path_name in [
        "root_dir",
        "data_dir",
        "log_dir",
        "ckpt_dir",
        "artifact_dir",
        "output_dir",
    ]:
        logger.info(
            f"{path_name.replace('_', ' ').capitalize()}: {cfg.paths[path_name]}"
        )

    # Initialize DataModule and Model
    logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    # Check GPU availability and set seed for reproducibility
    logger.info("GPU available" if torch.cuda.is_available() else "No GPU available")
    L.seed_everything(cfg.seed, workers=True)

    # Set up callbacks, loggers, and Trainer
    callbacks = initialize_callbacks(cfg)
    logger.info(f"Callbacks: {callbacks}")
    loggers = initialize_loggers(cfg)
    logger.info(f"Loggers: {loggers}")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers
    )

    # Training phase
    train_metrics = {}
    if cfg.get("train"):
        clear_checkpoint_directory(cfg.paths.ckpt_dir)
        train_metrics = train_module(datamodule, model, trainer)
        (Path(cfg.paths.ckpt_dir) / "train_done.flag").write_text(
            "Training completed.\n"
        )

    # Testing phase
    test_metrics = {}
    if cfg.get("test"):
        test_metrics = run_test_module(cfg, datamodule, model, trainer)

    # Combine metrics and extract optimization metric
    all_metrics = {**train_metrics, **test_metrics}
    optimization_metric = all_metrics.get(cfg.get("optimization_metric"), 0.0)
    (
        logger.warning(
            f"Optimization metric '{cfg.get('optimization_metric')}' not found. Defaulting to 0."
        )
        if optimization_metric == 0.0
        else logger.info(f"Optimization metric: {optimization_metric}")
    )

    return optimization_metric


if __name__ == "__main__":
    setup_run_trainer()
