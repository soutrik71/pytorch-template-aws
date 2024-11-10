import hydra
from omegaconf import DictConfig, OmegaConf
import rootutils
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(".env"))

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


# Define a ModelCheckpoint class that takes in parameters as specified in cfg.callbacks.model_checkpoint
class ModelCheckpoint:
    def __init__(
        self,
        dirpath,
        filename,
        monitor,
        verbose=False,
        save_last=True,
        save_top_k=1,
        mode="max",
        auto_insert_metric_name=False,
        save_weights_only=False,
        every_n_train_steps=None,
        train_time_interval=None,
        every_n_epochs=None,
        save_on_train_epoch_end=None,
    ):
        self.dirpath = dirpath
        self.filename = filename
        self.monitor = monitor
        self.verbose = verbose
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.mode = mode
        self.auto_insert_metric_name = auto_insert_metric_name
        self.save_weights_only = save_weights_only
        self.every_n_train_steps = every_n_train_steps
        self.train_time_interval = train_time_interval
        self.every_n_epochs = every_n_epochs
        self.save_on_train_epoch_end = save_on_train_epoch_end

    def display(self):
        print("Initialized ModelCheckpoint with the following configuration:")
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")


# Define func4 to initialize the ModelCheckpoint class using cfg.callbacks.model_checkpoint
def func4(**kwargs):
    # Initialize ModelCheckpoint with the kwargs
    checkpoint = ModelCheckpoint(**kwargs)
    checkpoint.display()  # Display the configuration for confirmation


@hydra.main(config_path="../configs", config_name="train", version_base="1.1")
def hydra_test(cfg: DictConfig):
    # Print the full configuration
    print("Full Configuration:")

    # Call func4 with the model checkpoint configuration
    func4(**cfg.callbacks.model_checkpoint)


if __name__ == "__main__":
    hydra_test()
