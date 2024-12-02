from pathlib import Path
import requests
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from src.models.catdog_model_resnet import ResnetClassifier
from src.utils.logging_utils import setup_logger, task_wrapper, get_rich_progress
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv, find_dotenv
import rootutils
import time
from loguru import logger
from src.utils.aws_s3_services import S3Handler

# Load environment variables
load_dotenv(find_dotenv(".env"))

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root")


@task_wrapper
def load_image(image_path: str, image_size: int):
    """Load and preprocess an image."""
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return img, transform(img).unsqueeze(0)


@task_wrapper
def infer(model: torch.nn.Module, image_tensor: torch.Tensor, classes: list):
    """Perform inference on the provided image tensor."""
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    predicted_label = classes[predicted_class]
    confidence = probabilities[0][predicted_class].item()
    return predicted_label, confidence


@task_wrapper
def save_prediction_image(
    image: Image.Image, predicted_label: str, confidence: float, output_path: Path
):
    """Save the image with the prediction overlay."""
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Predicted: {predicted_label} (Confidence: {confidence:.2f})")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


@task_wrapper
def download_image(cfg: DictConfig):
    """Download an image from the web for inference."""
    url = "https://github.com/laxmimerit/dog-cat-full-dataset/raw/master/data/train/dogs/dog.1.jpg"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36",
    }
    response = requests.get(url, headers=headers, allow_redirects=True)
    if response.status_code == 200:
        image_path = Path(cfg.paths.root_dir) / "image.jpg"
        with open(image_path, "wb") as file:
            file.write(response.content)
        time.sleep(5)
        print(f"Image downloaded successfully as {image_path}!")
    else:
        logger.error(f"Failed to download image. Status code: {response.status_code}")


@hydra.main(config_path="../configs", config_name="infer", version_base="1.3")
def main_infer(cfg: DictConfig):
    # Print the configuration
    logger.info(OmegaConf.to_yaml(cfg))
    setup_logger(Path(cfg.paths.log_dir) / "infer.log")

    # Remove the train_done flag if it exists
    flag_file = Path(cfg.paths.ckpt_dir) / "train_done.flag"
    if flag_file.exists():
        flag_file.unlink()

    # download the model from S3
    s3_handler = S3Handler(bucket_name="deep-bucket-s3")
    s3_handler.download_folder(
        "checkpoints",
        "checkpoints",
    )
    # Load the trained model
    model = ResnetClassifier.load_from_checkpoint(checkpoint_path=cfg.ckpt_path)
    classes = cfg.labels

    # Download an image for inference
    download_image(cfg)

    # Load images from directory and perform inference
    image_files = [
        f
        for f in Path(cfg.paths.root_dir).iterdir()
        if f.suffix in {".jpg", ".jpeg", ".png"}
    ]

    with get_rich_progress() as progress:
        task = progress.add_task("[green]Processing images...", total=len(image_files))

        for image_file in image_files:
            img, img_tensor = load_image(image_file, cfg.data.image_size)
            predicted_label, confidence = infer(
                model, img_tensor.to(model.device), classes
            )
            output_file = (
                Path(cfg.paths.artifact_dir) / f"{image_file.stem}_prediction.png"
            )
            save_prediction_image(img, predicted_label, confidence, output_file)
            progress.advance(task)

            logger.info(f"Processed {image_file}: {predicted_label} ({confidence:.2f})")


if __name__ == "__main__":
    main_infer()
