from pathlib import Path
from typing import Union, Tuple, Optional, List
import os
import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
from loguru import logger


class CatDogImageDataModule(L.LightningDataModule):
    """DataModule for Cat and Dog Image Classification using ImageFolder."""

    def __init__(
        self,
        root_dir: Union[str, Path] = "data",
        data_dir: Union[str, Path] = "cats_and_dogs_filtered",
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_split: List[float] = [0.8, 0.2],
        pin_memory: bool = False,
        image_size: int = 224,
        url: str = "https://download.pytorch.org/tutorials/cats_and_dogs_filtered.zip",
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.pin_memory = pin_memory
        self.image_size = image_size
        self.url = url

        # Initialize variables for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """Download the dataset if it doesn't exist."""
        self.dataset_path = self.root_dir / self.data_dir
        if not self.dataset_path.exists():
            logger.info("Downloading and extracting dataset.")
            download_and_extract_archive(
                url=self.url, download_root=self.root_dir, remove_finished=True
            )
            logger.info("Download completed.")

    def setup(self, stage: Optional[str] = None):
        """Set up the train, validation, and test datasets."""

        self.prepare_data()

        train_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(0.5),  # Flip probability increased
                transforms.RandomRotation(5),  # Reduced rotation for stability
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        train_path = self.dataset_path / "train"
        test_path = self.dataset_path / "test"

        if stage == "fit" or stage is None:
            full_train_dataset = ImageFolder(root=train_path, transform=train_transform)
            self.class_names = full_train_dataset.classes
            train_size = int(self.train_val_split[0] * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                full_train_dataset, [train_size, val_size]
            )
            logger.info(
                f"Train/Validation split: {len(self.train_dataset)} train, {len(self.val_dataset)} validation images."
            )

        if stage == "test" or stage is None:
            self.test_dataset = ImageFolder(root=test_path, transform=test_transform)
            logger.info(f"Test dataset size: {len(self.test_dataset)} images.")

    def _create_dataloader(self, dataset, shuffle: bool = False) -> DataLoader:
        """Helper function to create a DataLoader."""
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.test_dataset)

    def get_class_names(self) -> List[str]:
        return self.class_names


if __name__ == "__main__":
    # Test the CatDogImageDataModule
    import hydra
    from omegaconf import DictConfig, OmegaConf
    import rootutils

    root = rootutils.setup_root(__file__, indicator=".project-root")

    @hydra.main(
        config_path=str(root / "configs"), version_base="1.3", config_name="train"
    )
    def test_datamodule(cfg: DictConfig):
        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
        datamodule = CatDogImageDataModule(
            root_dir=cfg.data.root_dir,
            data_dir=cfg.data.data_dir,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            train_val_split=cfg.data.train_val_split,
            pin_memory=cfg.data.pin_memory,
            image_size=cfg.data.image_size,
        )
        datamodule.setup(stage="fit")
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage="test")
        test_loader = datamodule.test_dataloader()
        class_names = datamodule.get_class_names()

        logger.info(f"Train loader: {len(train_loader)} batches")
        logger.info(f"Validation loader: {len(val_loader)} batches")
        logger.info(f"Test loader: {len(test_loader)} batches")
        logger.info(f"Class names: {class_names}")

    test_datamodule()
