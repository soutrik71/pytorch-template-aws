import requests
from urllib.request import urlopen
import base64
import os


def fetch_image(url):
    """
    Fetch image data from a URL.
    """
    return urlopen(url).read()


def encode_image_to_base64(img_data):
    """
    Encode image bytes to a base64 string.
    """
    return base64.b64encode(img_data).decode("utf-8")


def send_prediction_request(base64_image, server_url):
    """
    Send a single base64 image to the prediction API and retrieve predictions.
    """
    try:
        response = requests.post(f"{server_url}/predict", json={"image": base64_image})
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to the server: {e}")
        return None


def send_batch_prediction_request(base64_images, server_url):
    """
    Send a batch of base64 images to the prediction API and retrieve predictions.
    """
    try:
        response = requests.post(
            f"{server_url}/predict", json=[{"image": img} for img in base64_images]
        )
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to the server: {e}")
        return None


def main():
    # Server URL (default or from environment)
    server_url = os.getenv("SERVER_URL", "http://localhost:8080")

    # Example URLs for testing
    image_urls = [
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    ]

    # Fetch and encode images
    try:
        print("Fetching and encoding images...")
        base64_images = [encode_image_to_base64(fetch_image(url)) for url in image_urls]
        print("Images fetched and encoded successfully.")
    except Exception as e:
        print(f"Error fetching or encoding images: {e}")
        return

    # Test single image prediction
    try:
        print("\n--- Single Image Prediction ---")
        single_response = send_prediction_request(base64_images[0], server_url)
        if single_response and single_response.status_code == 200:
            predictions = single_response.json().get("predictions", [])
            if predictions:
                print("Top 5 Predictions:")
                for pred in predictions:
                    print(f"{pred['label']}: {pred['probability']:.2%}")
            else:
                print("No predictions returned.")
        elif single_response:
            print(f"Error: {single_response.status_code}")
            print(single_response.text)
    except Exception as e:
        print(f"Error sending single prediction request: {e}")


if __name__ == "__main__":
    main()
