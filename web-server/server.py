# web_server.py
import os
import zlib
import json
import logging
import httpx
import redis.asyncio as redis
import socket
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fasthtml.common import (
    Html,
    Head,
    Title,
    Body,
    Form as FastHtmlForm,
    Input,
    Div,
    Button,
    H1,
    H2,
    P,
    to_xml,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - WebServer - %(levelname)s - %(message)s"
)
logger = logging.getLogger("WebServer")

app = FastAPI(title="CatDog Web Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://model-server:8000")

redis_pool = None


@app.on_event("startup")
async def startup_event():
    global redis_pool
    redis_pool = redis.ConnectionPool(
        host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True
    )
    logger.info("Web server startup complete.")


@app.on_event("shutdown")
async def shutdown_event():
    await redis_pool.disconnect()
    logger.info("Web server shutdown complete.")


def get_redis():
    return redis.Redis(connection_pool=redis_pool)


async def check_cache(image: bytes):
    cache = get_redis()
    hash = zlib.adler32(image)
    logger.info(f"Checking cache for hash: {hash}")
    return await cache.get(hash)


@app.get("/", response_class=HTMLResponse)
async def ui_home():
    content = Html(
        Head(Title("Cat vs Dog Classifier")),
        Body(
            Div(
                H1("Cat vs Dog Classifier", cls="text-4xl font-bold text-center my-6"),
                P(
                    "Upload an image of a cat or dog, and our AI model will classify it!",
                    cls="text-lg text-center mb-6",
                ),
                Div(
                    FastHtmlForm(
                        Input(
                            type="file",
                            name="file",
                            accept="image/*",
                            cls="block w-full text-sm text-gray-800 border border-gray-300 rounded-lg p-2",
                        ),
                        Button(
                            "Classify",
                            type="submit",
                            cls="mt-4 bg-blue-500 text-white font-bold px-4 py-2 rounded",
                        ),
                        action="/classify",
                        enctype="multipart/form-data",
                        method="post",
                        cls="flex flex-col gap-4",
                    ),
                    cls="container mx-auto max-w-md",
                ),
            ),
            cls="min-h-screen bg-gray-100 flex flex-col items-center justify-center",
        ),
    )
    return to_xml(content)


@app.post("/classify", response_class=HTMLResponse)
async def classify(file: UploadFile = File(...)):
    try:
        image = await file.read()

        # Check cache
        cached_result = await check_cache(image)
        if cached_result:
            logger.info("Cache hit for the uploaded image.")
            cached_result = json.loads(cached_result)
            return HTMLResponse(
                content=f"<h1 class='text-2xl font-bold text-center'>Prediction: {cached_result['label']} ({cached_result['confidence']:.2%})</h1>"
            )

        # Call the model server
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MODEL_SERVER_URL}/infer", files={"image": image}
            )
            response.raise_for_status()
            result = response.json()

            # Cache the result
            cache = get_redis()
            hash = zlib.adler32(image)
            await cache.set(hash, json.dumps(result))
            logger.info("Result cached successfully.")

            # Display result on UI
            return HTMLResponse(
                content=f"<h1 class='text-2xl font-bold text-center'>Prediction: {result['label']} ({result['confidence']:.2%})</h1>"
            )
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail="Classification failed")


@app.get("/health", response_model=dict)
async def health_check():
    redis_connected = False
    model_server_connected = False
    model_server_health = None

    # Test Redis connectivity
    try:
        redis_client = get_redis()
        redis_connected = await redis_client.ping()
        logger.info("Redis connection successful.")
    except Exception as e:
        logger.error(f"Redis health check failed: {str(e)}")

    # Test Model Server health
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{MODEL_SERVER_URL}/health")
            response.raise_for_status()
            model_server_health = response.json()
            model_server_connected = True
            logger.info("Model server health check successful.")
    except Exception as e:
        logger.error(f"Model server health check failed: {str(e)}")

    health_status = {
        "status": (
            "healthy" if redis_connected and model_server_connected else "degraded"
        ),
        "redis": {
            "host": REDIS_HOST,
            "port": REDIS_PORT,
            "connected": redis_connected,
        },
        "model_server": {
            "url": MODEL_SERVER_URL,
            "connected": model_server_connected,
            "health": model_server_health,
        },
        "hostname": socket.gethostname(),
    }

    logger.info(f"Health check status: {health_status['status']}")
    return health_status


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000, reload=True)
