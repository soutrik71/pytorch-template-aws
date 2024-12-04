from typing import List
import os
from src.utils.multirun_op import multirun_artifact_producer
import hydra
from omegaconf import DictConfig
from loguru import logger
from dotenv import load_dotenv, find_dotenv
import rootutils

# Load environment variables
load_dotenv(find_dotenv(".env"))

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root")


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def create_artifacts(cfg: DictConfig):
    base_path = os.path.join(cfg.paths.log_dir, "train", "runs")
    logger.info(
        f"Base path: {base_path} and artifact directory: {cfg.paths.artifact_dir}"
    )
    multirun_artifact_producer(base_path, cfg.paths.artifact_dir)


if __name__ == "__main__":
    create_artifacts()
