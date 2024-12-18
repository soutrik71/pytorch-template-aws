import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from torchvision import transforms
from src.models.catdog_model_resnet import ResnetClassifier
from src.utils.aws_s3_services import S3Handler
from src.utils.logging_utils import setup_logger
from loguru import logger
import rootutils
import os

# Load environment variables and configure logger
log_dir = "/tmp/logs"
os.makedirs(log_dir, exist_ok=True)
setup_logger(Path(log_dir) / "gradio_app.log")
# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root")


class ImageClassifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classes = cfg.labels

        # Download and load model from S3
        logger.info("Downloading model from S3...")
        s3_handler = S3Handler(bucket_name="deep-bucket-s3")
        s3_handler.download_folder("checkpoints", "checkpoints")

        logger.info("Loading model checkpoint...")
        self.model = ResnetClassifier.load_from_checkpoint(
            checkpoint_path=cfg.ckpt_path
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Image transform
        self.transform = transforms.Compose(
            [
                transforms.Resize((cfg.data.image_size, cfg.data.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict(self, image):
        if image is None:
            return "No image provided.", None

        # Preprocess the image
        logger.info("Processing input image...")
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()

        predicted_label = self.classes[predicted_class_idx]
        logger.info(f"Prediction: {predicted_label} (Confidence: {confidence:.2f})")
        return predicted_label, confidence


def create_gradio_app(cfg):
    classifier = ImageClassifier(cfg)

    def classify_image(image):
        """Gradio interface function."""
        predicted_label, confidence = classifier.predict(image)
        if predicted_label:
            return f"Predicted: {predicted_label} (Confidence: {confidence:.2f})"
        return "Error during prediction."

    # Create Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Cat vs Dog Classifier
            Upload an image of a cat or a dog to classify it with confidence.
            """
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Input Image", type="pil", image_mode="RGB"
                )
                predict_button = gr.Button("Classify")
            with gr.Column():
                output_text = gr.Textbox(label="Prediction")

        # Define interaction
        predict_button.click(
            fn=classify_image, inputs=[input_image], outputs=[output_text]
        )

    return demo


# Hydra config wrapper for launching Gradio app
if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="configs", config_name="infer", version_base="1.3")
    def main(cfg: DictConfig):
        logger.info("Launching Gradio App...")
        demo = create_gradio_app(cfg)
        demo.launch(share=True, server_name="0.0.0.0", server_port=7860)

    main()
