"""
Train and evaluate a model using PyTorch Lightning with Optuna for hyperparameter optimization.
"""

import os
import shutil
from pathlib import Path
from typing import List
import torch
import lightning as L
from dotenv import load_dotenv, find_dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.logging_utils import setup_logger, task_wrapper
from loguru import logger
import rootutils
from lightning.pytorch.loggers import Logger
import optuna
from lightning.pytorch import Trainer

# Load environment variables
load_dotenv(find_dotenv(".env"))

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root")


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiate and return a list of loggers from the configuration."""
    loggers_ls: List[Logger] = []

    if not logger_cfg or isinstance(logger_cfg, bool):
        logger.warning("No valid logger configs found! Skipping..")
        return loggers_ls

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            logger.info(f"Instantiating logger <{lg_conf._target_}>")
            try:
                loggers_ls.append(hydra.utils.instantiate(lg_conf))
            except Exception as e:
                logger.error(f"Failed to instantiate logger {lg_conf}: {e}")
    return loggers_ls


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
    """Train the model, return validation accuracy for each epoch."""
    logger.info("Starting training with custom pruning")

    trainer.fit(model, data_module)
    val_accuracies = []

    for epoch in range(trainer.current_epoch):
        val_acc = trainer.callback_metrics.get("val_acc")
        if val_acc is not None:
            val_accuracies.append(val_acc.item())
            logger.info(f"Epoch {epoch}: val_acc={val_acc}")

    return val_accuracies


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


def objective(trial: optuna.trial.Trial, cfg: DictConfig):
    """Objective function for Optuna hyperparameter tuning."""

    # Sample hyperparameters for the model
    cfg.model.embed_dim = trial.suggest_categorical("embed_dim", [64, 128, 256])
    cfg.model.depth = trial.suggest_int("depth", 2, 6)
    cfg.model.lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    cfg.model.mlp_ratio = trial.suggest_float("mlp_ratio", 1.0, 4.0)

    # Initialize data module and model
    data_module: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    # Set up logger
    loggers = instantiate_loggers(cfg.logger)

    # Trainer configuration without pruning callback
    trainer = Trainer(**cfg.trainer, logger=loggers)

    # Clear checkpoint directory
    clear_checkpoint_directory(cfg.paths.ckpt_dir)

    # Train and get val_acc for each epoch
    val_accuracies = train_module(data_module, model, trainer)

    # Report validation accuracy and prune if necessary
    for epoch, val_acc in enumerate(val_accuracies):
        trial.report(val_acc, step=epoch)

        # Check if the trial should be pruned at this epoch
        if trial.should_prune():
            logger.info(f"Pruning trial at epoch {epoch}")
            raise optuna.TrialPruned()

    # Return the final validation accuracy as the objective metric
    return val_accuracies[-1] if val_accuracies else 0.0


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def setup_trainer(cfg: DictConfig):
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    setup_logger(
        Path(cfg.paths.log_dir)
        / ("train.log" if cfg.task_name == "train" else "eval.log")
    )

    if cfg.get("train", False):
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(
            direction="maximize", pruner=pruner, study_name="pytorch_lightning_optuna"
        )
        study.optimize(
            lambda trial: objective(trial, cfg), n_trials=3, show_progress_bar=True
        )

        # Log best trial results
        best_trial = study.best_trial
        logger.info(f"Best trial number: {best_trial.number}")
        logger.info(f"Best trial value (val_acc): {best_trial.value}")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")

    if cfg.get("test", False):
        data_module: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
        model: L.LightningModule = hydra.utils.instantiate(cfg.model)
        trainer = Trainer(**cfg.trainer, logger=instantiate_loggers(cfg.logger))
        test_metrics = run_test_module(cfg, data_module, model, trainer)
        logger.info(f"Test metrics: {test_metrics}")

    return cfg.model if not cfg.get("test", False) else test_metrics


if __name__ == "__main__":
    setup_trainer()
