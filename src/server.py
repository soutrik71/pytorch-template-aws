import torch
from PIL import Image
import io
import litserve as lit
import base64
from torchvision import transforms
from src.models.catdog_model import ViTTinyClassifier
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv, find_dotenv
import rootutils
from loguru import logger
from src.utils.logging_utils import setup_logger
from pathlib import Path

# Load environment variables
load_dotenv(find_dotenv(".env"))

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root")
logger.info(f"Root directory set to: {root}")


class ImageClassifierAPI(lit.LitAPI):
    def __init__(self, cfg: DictConfig):
        """
        Initialize the API with Hydra configuration.
        """
        super().__init__()
        self.cfg = cfg

        # Validate required config keys
        required_keys = ["ckpt_path", "data.image_size", "labels"]
        missing_keys = [key for key in required_keys if not OmegaConf.select(cfg, key)]
        if missing_keys:
            logger.error(f"Missing required config keys: {missing_keys}")
            raise ValueError(f"Missing required config keys: {missing_keys}")
        logger.info(f"Configuration validated: {OmegaConf.to_yaml(cfg)}")

    def setup(self, device):
        """Initialize the model and necessary components."""
        self.device = device
        logger.info("Setting up the model and components.")

        # Log the configuration for debugging
        logger.debug(f"Configuration passed to setup: {OmegaConf.to_yaml(self.cfg)}")

        # Load the model from checkpoint
        try:
            self.model = ViTTinyClassifier.load_from_checkpoint(
                checkpoint_path=self.cfg.ckpt_path
            )
            self.model = self.model.to(device).eval()
            logger.info("Model loaded and moved to device.")
        except FileNotFoundError:
            logger.error(f"Checkpoint file not found: {self.cfg.ckpt_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

        # Define transforms
        self.transforms = transforms.Compose(
            [
                transforms.Resize((self.cfg.data.image_size, self.cfg.data.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # Hard-coded mean
                    std=[0.229, 0.224, 0.225],  # Hard-coded std
                ),
            ]
        )
        logger.info("Transforms initialized.")

        # Load labels
        try:
            self.labels = self.cfg.labels
            logger.info(f"Labels loaded: {self.labels}")
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            raise ValueError("Failed to load labels from the configuration.")

    def decode_request(self, request):
        """Handle both single and batch inputs."""
        # logger.info(f"decode_request received: {request}")
        if not isinstance(request, dict) or "image" not in request:
            logger.error(
                "Invalid request format. Expected a dictionary with key 'image'."
            )
            raise ValueError(
                "Invalid request format. Expected a dictionary with key 'image'."
            )
        return request["image"]

    def batch(self, inputs):
        """Batch process images."""
        # logger.info(f"batch received inputs: {inputs}")
        if not isinstance(inputs, list):
            raise ValueError("Input to batch must be a list.")

        batch_tensors = []
        try:
            for image_bytes in inputs:
                if not isinstance(image_bytes, str):  # Ensure input is a base64 string
                    raise ValueError(
                        f"Input must be a base64-encoded string, got: {type(image_bytes)}"
                    )

                # Decode base64 string to bytes
                img_bytes = base64.b64decode(image_bytes)

                # Convert bytes to PIL Image
                try:
                    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception as img_error:
                    logger.error(f"Failed to process image: {img_error}")
                    raise

                # Apply transforms and add to batch
                tensor = self.transforms(image)
                batch_tensors.append(tensor)

            return torch.stack(batch_tensors).to(self.device)
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            raise ValueError("Failed to decode and process the images.")

    def predict(self, x):
        """Make predictions on the input batch."""
        with torch.inference_mode():
            outputs = self.model(x)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        logger.info("Prediction completed.")
        return probabilities

    def unbatch(self, output):
        """Unbatch the output."""
        return [output[i] for i in range(output.size(0))]

    def encode_response(self, output):
        """Convert model output to API response for batches."""
        try:
            probs, indices = torch.topk(output, k=1)
            responses = {
                "predictions": [
                    {
                        "label": self.labels[idx.item()],
                        "probability": prob.item(),
                    }
                    for prob, idx in zip(probs, indices)
                ]
            }
            logger.info("Batch response successfully encoded.")
            return responses
        except Exception as e:
            logger.error(f"Error encoding batch response: {e}")
            raise ValueError("Failed to encode the batch response.")


@hydra.main(config_path="../configs", config_name="infer", version_base="1.3")
def main(cfg: DictConfig):
    # Initialize loguru
    setup_logger(Path(cfg.paths.log_dir) / "infer.log")
    logger.info("Starting the Image Classifier API server.")

    # Log configuration
    logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")

    # Create the API instance with the Hydra config
    api = ImageClassifierAPI(cfg)

    # Configure the server
    server = lit.LitServer(
        api,
        accelerator=cfg.server.accelerator,
        max_batch_size=cfg.server.max_batch_size,
        batch_timeout=cfg.server.batch_timeout,
        devices=cfg.server.devices,
        workers_per_device=cfg.server.workers_per_device,
    )
    server.run(port=cfg.server.port)


if __name__ == "__main__":
    main()
