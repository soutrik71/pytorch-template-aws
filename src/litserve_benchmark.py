import base64
import concurrent.futures
import time
import numpy as np
import requests
import psutil
from urllib.request import urlopen
import matplotlib.pyplot as plt
from loguru import logger
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Try importing `gpustat` for GPU monitoring
try:
    import gpustat

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Constants
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8080")
TEST_IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"


def fetch_and_prepare_payload():
    """
    Fetch the test image and prepare a base64 payload.
    """
    try:
        img_data = urlopen(TEST_IMAGE_URL).read()
        return base64.b64encode(img_data).decode("utf-8")
    except Exception as e:
        logger.info(f"Error fetching the image: {e}")
        return None


def send_request(payload, batch=False):
    """
    Send a single or batch request and measure response time.
    """
    start_time = time.time()
    endpoint = f"{SERVER_URL}/predict"
    try:
        if batch:
            response = requests.post(endpoint, json=[{"image": img} for img in payload])
        else:
            response = requests.post(endpoint, json={"image": payload})
        response_time = time.time() - start_time
        predictions = response.json() if response.status_code == 200 else None
        return response_time, response.status_code, predictions
    except Exception as e:
        logger.info(f"Error sending request: {e}")
        return None, None, None


def get_system_metrics():
    """
    Get current CPU and GPU usage.
    """
    metrics = {"cpu_usage": psutil.cpu_percent(0.1)}
    if GPU_AVAILABLE:
        try:
            gpu_stats = gpustat.GPUStatCollection.new_query()
            metrics["gpu_usage"] = sum([gpu.utilization for gpu in gpu_stats.gpus])
        except Exception:
            metrics["gpu_usage"] = -1
    else:
        metrics["gpu_usage"] = -1
    return metrics


def benchmark_api(num_requests=100, concurrency_level=10, batch=False):
    """
    Benchmark the API server.
    """
    payload = fetch_and_prepare_payload()
    if not payload:
        logger.info("Error preparing payload. Benchmark aborted.")
        return

    payloads = [payload] * num_requests if batch else [payload]
    system_metrics = []
    response_times = []
    status_codes = []
    predictions = []

    # Start benchmark timer
    start_benchmark_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency_level
    ) as executor:
        futures = [
            executor.submit(send_request, payloads if batch else payload, batch)
            for _ in range(num_requests)
        ]
        while any(not f.done() for f in futures):
            system_metrics.append(get_system_metrics())
            time.sleep(0.1)

        for future in futures:
            result = future.result()
            if result:
                response_time, status_code, prediction = result
                response_times.append(response_time)
                status_codes.append(status_code)
                predictions.append(prediction)

    # Stop benchmark timer
    total_benchmark_time = time.time() - start_benchmark_time

    avg_cpu = np.mean([m["cpu_usage"] for m in system_metrics])
    avg_gpu = np.mean([m["gpu_usage"] for m in system_metrics]) if GPU_AVAILABLE else -1

    success_rate = (status_codes.count(200) / num_requests) * 100 if status_codes else 0
    avg_response_time = np.mean(response_times) * 1000 if response_times else 0  # ms
    requests_per_second = num_requests / total_benchmark_time

    logger.info("\n--- Sample Predictions ---")
    for i, prediction in enumerate(
        predictions[:5]
    ):  # Show predictions for the first 5 requests
        logger.info(f"Request {i + 1}: {prediction}")

    return {
        "total_requests": num_requests,
        "concurrency_level": concurrency_level,
        "total_time": total_benchmark_time,
        "avg_response_time": avg_response_time,
        "success_rate": success_rate,
        "requests_per_second": requests_per_second,
        "avg_cpu_usage": avg_cpu,
        "avg_gpu_usage": avg_gpu,
    }


def run_benchmarks():
    """
    Run comprehensive benchmarks and create separate plots for CPU and GPU usage.
    """
    concurrency_levels = [1, 8, 16, 32]
    metrics = []

    logger.info("Running API benchmarks...")
    for concurrency in concurrency_levels:
        logger.info(f"\nTesting concurrency level: {concurrency}")
        result = benchmark_api(
            num_requests=50, concurrency_level=concurrency, batch=False
        )
        if result:
            metrics.append(result)
            logger.info(
                f"Concurrency {concurrency}: "
                f"{result['requests_per_second']:.2f} reqs/sec, "
                f"CPU: {result['avg_cpu_usage']:.1f}%, "
                f"GPU: {result['avg_gpu_usage']:.1f}%"
            )

    # Generate CPU Usage Plot
    plt.figure(figsize=(10, 5))
    plt.plot(
        concurrency_levels,
        [m["avg_cpu_usage"] for m in metrics],
        "b-o",
        label="CPU Usage",
    )
    plt.xlabel("Concurrency Level")
    plt.ylabel("CPU Usage (%)")
    plt.title("CPU Usage vs. Concurrency Level")
    plt.grid(True)
    plt.savefig("artifacts/cpu_usage.png")
    logger.info("CPU usage plot saved as 'cpu_usage.png'.")

    # Generate GPU Usage Plot
    if GPU_AVAILABLE:
        plt.figure(figsize=(10, 5))
        plt.plot(
            concurrency_levels,
            [m["avg_gpu_usage"] for m in metrics],
            "g-o",
            label="GPU Usage",
        )
        plt.xlabel("Concurrency Level")
        plt.ylabel("GPU Usage (%)")
        plt.title("GPU Usage vs. Concurrency Level")
        plt.grid(True)
        plt.savefig("artifacts/gpu_usage.png")
        logger.info("GPU usage plot saved as 'gpu_usage.png'.")


if __name__ == "__main__":
    run_benchmarks()
