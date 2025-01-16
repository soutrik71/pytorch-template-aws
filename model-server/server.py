# model_server.py
import os
import io
import zlib
import logging
import json
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from fastapi import FastAPI, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.models.catdog_model_resnet import ResnetClassifier
import hydra
from omegaconf import DictConfig
import redis.asyncio as redis

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - ModelServer - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ModelServer")

app = FastAPI(title="CatDog Model Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = None
LABELS = None
IMAGE_SIZE = None
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "redispassword")
redis_pool = None


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CatDogClassifier:
    def __init__(self, model_path, labels, image_size):
        self.device = get_device()
        self.model = self.load_model(model_path)
        self.model.eval()
        self.labels = labels
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def load_model(self, path):
        if not Path(path).exists():
            logger.error(f"Model file not found at {path}")
            raise FileNotFoundError(f"Model file not found at {path}")
        logger.info(f"Loading model from {path}")
        return ResnetClassifier.load_from_checkpoint(path).to(self.device)

    def predict(self, image: Image.Image):
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()
        return self.labels[predicted_idx], confidence


classifier = None


def get_redis():
    return redis.Redis(connection_pool=redis_pool)


@app.on_event("startup")
async def startup_event():
    global classifier, MODEL_PATH, LABELS, IMAGE_SIZE, redis_pool

    # Load Hydra Config
    with hydra.initialize(config_path="configs", version_base="1.3"):
        cfg = hydra.compose(config_name="infer")

    MODEL_PATH = cfg.ckpt_path
    LABELS = cfg.labels
    IMAGE_SIZE = cfg.data.image_size

    logger.info("Starting CatDogClassifier...")
    classifier = CatDogClassifier(MODEL_PATH, LABELS, IMAGE_SIZE)

    redis_pool = redis.ConnectionPool(
        host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True
    )
    logger.info("Redis connection pool initialized.")


@app.on_event("shutdown")
async def shutdown_event():
    await redis_pool.disconnect()
    logger.info("Redis connection pool closed.")


@app.post("/infer", response_model=dict)
async def infer(image: bytes = File(...)):
    try:
        cache = get_redis()
        hash = zlib.adler32(image)

        # Check if result is cached
        cached_result = await cache.get(hash)
        if cached_result:
            logger.info("Cache hit for image")
            return json.loads(cached_result)

        # Perform inference
        img = Image.open(io.BytesIO(image)).convert("RGB")
        predicted_label, confidence = classifier.predict(img)

        # Cache result
        result = {"label": predicted_label, "confidence": confidence}
        await cache.set(hash, json.dumps(result))
        logger.info("Prediction cached successfully")
        return result

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")


@app.get("/health", response_model=dict)
async def health_check():
    redis_connected = False
    try:
        redis_client = get_redis()
        redis_connected = await redis_client.ping()
        logger.info("Redis connection successful.")
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")

    status = {
        "status": "healthy" if classifier and redis_connected else "degraded",
        "model_loaded": classifier is not None,
        "redis": {
            "host": REDIS_HOST,
            "port": REDIS_PORT,
            "connected": redis_connected,
        },
        "device": str(get_device()),
    }
    logger.info(f"Health check status: {status['status']}")
    return status


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
