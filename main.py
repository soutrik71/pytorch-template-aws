import io
import os
import base64
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from fastapi import FastAPI, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fasthtml.common import (
    Html,
    Head,
    Title,
    Body,
    Div,
    Form,
    Input,
    P,
    Button,
    Img,
    to_xml,
    Script,
)
from shad4fast import (
    ShadHead,
    Card,
    CardHeader,
    CardTitle,
    CardContent,
    Badge,
    Progress,
    Alert,
    AlertTitle,
    AlertDescription,
)
from loguru import logger
from src.models.catdog_model_resnet import ResnetClassifier
from src.utils.aws_s3_services import S3Handler
from dotenv import load_dotenv, find_dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
from typing import Annotated
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root")

# Load environment variables
load_dotenv(find_dotenv())

# Initialize FastAPI app
app = FastAPI(
    title="Cat vs Dog Classifier",
    description="Classify whether an image is of a cat or a dog.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Logging setup
log_dir = "/tmp/logs"
Path(log_dir).mkdir(parents=True, exist_ok=True)
logger.add(f"{log_dir}/inference.log", rotation="1 MB", level="INFO", enqueue=False)

# Global model and config variables
classifier = None
cfg = None


# Classifier Class
class CatDogClassifier:
    def __init__(self, model_path: str, labels: list, image_size: int):
        self.model_path = model_path
        self.labels = labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Inference will run on device: {self.device}")

        # Load model
        self.model = self.load_model()
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load_model(self):
        if not Path(self.model_path).exists():
            logger.error(f"Model not found: {self.model_path}")
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        logger.info(f"Loading model from: {self.model_path}")
        return ResnetClassifier.load_from_checkpoint(self.model_path).to(self.device)

    def predict(self, image: Image.Image):
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()
        return self.labels[predicted_idx], confidence


# @app.on_event("startup")
# async def startup_event():
#     global classifier, cfg

#     # Load Hydra Config
#     with hydra.initialize(config_path="configs", version_base="1.3"):
#         cfg = hydra.compose(config_name="infer")

#     model_path = cfg.ckpt_path
#     labels = cfg.labels
#     image_size = cfg.data.image_size

#     # Download model if not exists
#     if not Path(model_path).exists():
#         logger.info("Downloading model from S3...")
#         s3_handler = S3Handler(bucket_name="deep-bucket-s3")
#         s3_handler.download_folder("checkpoints", cfg.paths.ckpt_dir)

#     logger.info("Initializing Cat/Dog Classifier...")
#     classifier = CatDogClassifier(model_path, labels, image_size)


@app.on_event("startup")
async def startup_event():
    global classifier, cfg

    # Set cache paths to writable /tmp directory
    os.environ["HF_HOME"] = "/tmp/huggingface"  # HuggingFace Hub cache
    os.environ["TORCH_HOME"] = "/tmp/torch"  # PyTorch cache
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"  # Matplotlib cache

    # Ensure writable directories exist
    for cache_dir in ["/tmp/huggingface", "/tmp/torch", "/tmp/matplotlib"]:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Load Hydra Config
    with hydra.initialize(config_path="configs", version_base="1.3"):
        cfg = hydra.compose(config_name="infer")

    model_path = cfg.ckpt_path
    labels = cfg.labels
    image_size = cfg.data.image_size

    # Download model if not exists
    if not Path(model_path).exists():
        logger.info("Downloading model from S3...")
        s3_handler = S3Handler(bucket_name="deep-bucket-s3")
        s3_handler.download_folder("checkpoints", cfg.paths.ckpt_dir)

    logger.info("Initializing Cat/Dog Classifier...")
    classifier = CatDogClassifier(model_path, labels, image_size)


@app.get("/", response_class=HTMLResponse)
async def ui_home():
    """
    Serve the FastHTML-based UI for image upload and Cat/Dog prediction.
    """
    try:
        content = Html(
            Head(
                Title("Cat vs Dog Classifier"),
                ShadHead(tw_cdn=True, theme_handle=True),
                Script(
                    src="https://unpkg.com/htmx.org@2.0.3",
                    integrity="sha384-0895/pl2MU10Hqc6jd4RvrthNlDiE9U1tWmX7WRESftEDRosgxNsQG/Ze9YMRzHq",
                    crossorigin="anonymous",
                ),
            ),
            Body(
                Div(
                    Div(
                        Div(
                            CardHeader(
                                CardTitle(
                                    "Cat vs Dog Classifier 🐱🐶",
                                    cls="text-white text-2xl font-bold",
                                ),
                                Badge(
                                    "AI Powered",
                                    cls="bg-yellow-400 text-black px-3 py-1 rounded-full",
                                ),
                            ),
                            cls="flex items-center justify-between mb-4",
                        ),
                        CardContent(
                            Form(
                                Div(
                                    Input(
                                        type="file",
                                        name="file",
                                        accept="image/*",
                                        cls="block w-full text-sm text-gray-200 border border-gray-500 rounded-lg bg-gray-800 focus:ring-yellow-500 focus:border-yellow-500",
                                    ),
                                    Button(
                                        "Classify Image 🐾",
                                        type="submit",
                                        cls="mt-4 bg-yellow-500 text-black font-bold px-4 py-2 rounded-lg hover:bg-yellow-400 transition",
                                    ),
                                    cls="space-y-4",
                                ),
                                enctype="multipart/form-data",
                                hx_post="/classify",
                                hx_target="#result",
                            ),
                        ),
                        Div(id="result", cls="mt-6"),
                        cls="p-6 bg-gray-900 shadow-lg rounded-lg",
                    ),
                    cls="container mx-auto mt-20 max-w-lg",
                ),
                cls="bg-black text-white min-h-screen flex items-center justify-center",
            ),
        )
        return to_xml(content)
    except Exception as e:
        logger.error(f"Error rendering UI: {e}")
        error_alert = Html(
            Body(
                Alert(
                    AlertTitle("Rendering Error"),
                    AlertDescription(str(e)),
                    variant="destructive",
                    cls="mt-4",
                )
            )
        )
        return to_xml(error_alert)


@app.post("/classify", response_class=HTMLResponse)
async def classify_image(file: Annotated[bytes, File()]):
    try:
        image = Image.open(io.BytesIO(file)).convert("RGB")
        predicted_label, confidence = classifier.predict(image)

        results = Div(
            Badge(
                f"Prediction: {predicted_label} ({confidence:.1%})",
                cls="w-full text-lg bg-green-500 text-black font-bold px-4 py-2 rounded-lg",
            ),
            Progress(
                value=int(confidence * 100), cls="mt-4 h-4 rounded-lg bg-gray-700"
            ),
            Img(
                src=f"data:image/png;base64,{base64.b64encode(file).decode('utf-8')}",
                cls="w-32 h-32 rounded-full mt-4 mx-auto",
            ),
            cls="text-center mt-6",
        )
        return to_xml(results)

    except Exception as e:
        logger.error(f"Error during classification: {e}")
        error_alert = Alert(
            AlertTitle("Error ❌"),
            AlertDescription(str(e)),
            variant="destructive",
            cls="mt-4",
        )
        return to_xml(error_alert)


@app.get("/health")
async def health_check():
    return JSONResponse(
        content={"status": "healthy", "model_loaded": classifier is not None}
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server with FastHTML...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True,
        workers=1,
    )
