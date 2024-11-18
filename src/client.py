from loguru import logger
import requests
from urllib.request import urlopen
import base64
import os


def fetch_image(url):
    """
    Fetch image data from a URL.
    """
    try:
        return urlopen(url).read()
    except Exception as e:
        logger.error(f"Failed to fetch image from {url}: {e}")
        raise


def encode_image_to_base64(img_data):
    """
    Encode image bytes to a base64 string.
    """
    try:
        return base64.b64encode(img_data).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to encode image to base64: {e}")
        raise


def send_prediction_request(base64_image, server_urls):
    """
    Send a single base64 image to the prediction API and retrieve predictions.
    Tries multiple server URLs in order.
    """
    for server_url in server_urls:
        try:
            logger.info(f"Attempting to send prediction request to {server_url}...")
            response = requests.post(
                f"{server_url}/predict", json={"image": base64_image}
            )
            if response.status_code == 200:
                logger.info(f"Successfully connected to {server_url}")
                return response
            else:
                logger.warning(
                    f"Server at {server_url} returned status code {response.status_code}"
                )
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to the server at {server_url}: {e}")
    logger.error("Failed to connect to any server.")
    return None


def main():
    # Server URLs to try
    server_url_env = os.getenv("SERVER_URL", "http://localhost:8080")
    server_urls = [server_url_env]

    # Example URLs for testing
    image_urls = [
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    ]

    # Fetch and encode images
    try:
        logger.info("Fetching and encoding images...")
        base64_images = [encode_image_to_base64(fetch_image(url)) for url in image_urls]
        logger.info("Images fetched and encoded successfully.")
    except Exception as e:
        logger.error(f"Error fetching or encoding images: {e}")
        return

    # Test single image prediction
    try:
        logger.info("--- Single Image Prediction ---")
        single_response = send_prediction_request(base64_images[0], server_urls)
        if single_response and single_response.status_code == 200:
            predictions = single_response.json().get("predictions", [])
            if predictions:
                logger.info("Top Predictions:")
                for pred in predictions:
                    logger.info(f"{pred['label']}: {pred['probability']:.2%}")
            else:
                logger.warning("No predictions returned.")
        elif single_response:
            logger.error(f"Error: {single_response.status_code}")
            logger.error(single_response.text)
        else:
            logger.error("Failed to get a response from any server.")
    except Exception as e:
        logger.error(f"Error sending single prediction request: {e}")


if __name__ == "__main__":
    main()
