import hydra
from omegaconf import DictConfig, OmegaConf
import rootutils
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(".env"))

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


# Define separate functions for each sub-configuration
def print_data(data_cfg: DictConfig):
    print("Data Configuration:")
    print(OmegaConf.to_yaml(data_cfg))


def print_model(model_cfg: DictConfig):
    print("Model Configuration:")
    print(OmegaConf.to_yaml(model_cfg))


def print_callbacks(callbacks_cfg: DictConfig):
    print("Callbacks Configuration:")
    print(OmegaConf.to_yaml(callbacks_cfg))


def print_logger(logger_cfg: DictConfig):
    print("Logger Configuration:")
    print(OmegaConf.to_yaml(logger_cfg))


def print_trainer(trainer_cfg: DictConfig):
    print("Trainer Configuration:")
    print(OmegaConf.to_yaml(trainer_cfg))


def print_paths(paths_cfg: DictConfig):
    print("Paths Configuration:")
    print(OmegaConf.to_yaml(paths_cfg))


@hydra.main(config_path="../configs", config_name="train", version_base="1.1")
def hydra_test(cfg: DictConfig):
    # Print the full configuration
    print("Full Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Call each function with the corresponding sub-configuration
    print_data(cfg.data)
    print_model(cfg.model)
    print_callbacks(cfg.callbacks)
    print_logger(cfg.logger)
    print_trainer(cfg.trainer)
    print_paths(cfg.paths)


if __name__ == "__main__":
    hydra_test()
