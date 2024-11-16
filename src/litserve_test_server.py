import torch
import timm
from PIL import Image
import io
import litserve as lit
import base64
import requests
import logging


class ImageClassifierAPI(lit.LitAPI):
    def setup(self, device):
        """Initialize the model and necessary components."""
        self.device = device
        logging.info("Setting up the model and components.")

        # Create and load the model
        self.model = timm.create_model("resnet50.a1_in1k", pretrained=True)
        self.model = self.model.to(device).eval()

        # Disable gradients to save memory
        with torch.no_grad():
            data_config = timm.data.resolve_model_data_config(self.model)
            self.transforms = timm.data.create_transform(
                **data_config, is_training=False
            )

        # Load labels
        url = "https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt"
        try:
            self.labels = requests.get(url).text.strip().split("\n")
            logging.info("Labels loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load labels: {e}")
            self.labels = []

    def decode_request(self, request):
        """Handle both single and batch inputs."""
        logging.info(f"decode_request received: {request}")
        if isinstance(request, dict):
            return request["image"]

    def batch(self, inputs):
        """Batch process images."""
        logging.info(f"batch received inputs: {inputs}")
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
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                # Apply transforms and add to batch
                tensor = self.transforms(image)
                batch_tensors.append(tensor)

            return torch.stack(batch_tensors).to(self.device)
        except Exception as e:
            logging.error(f"Error decoding image: {e}")
            raise ValueError("Failed to decode and process the images.")

    @torch.no_grad()
    def predict(self, x):
        """Make predictions on the input batch."""
        outputs = self.model(x)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        logging.info("Prediction completed.")
        return probabilities

    def unbatch(self, output):
        """Unbatch the output."""
        return [output[i] for i in range(output.size(0))]

    def encode_response(self, output):
        """Convert model output to API response for batches."""
        try:
            probs, indices = torch.topk(output, k=5)
            responses = {
                "predictions": [
                    {
                        "label": self.labels[idx.item()],
                        "probability": prob.item(),
                    }
                    for prob, idx in zip(probs, indices)
                ]
            }
            logging.info("Batch response successfully encoded.")
            return responses
        except Exception as e:
            logging.error(f"Error encoding batch response: {e}")
            raise ValueError("Failed to encode the batch response.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("Starting the Image Classifier API server.")

    api = ImageClassifierAPI()

    # Configure server with optimal settings
    server = lit.LitServer(
        api, accelerator="auto", max_batch_size=16, batch_timeout=0.01
    )
    server.run(port=8080)
