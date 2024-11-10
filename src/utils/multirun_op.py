import os
import glob
import yaml
import pandas as pd
import json
from datetime import datetime
from loguru import logger


def multirun_artifact_producer(base_path: str, output_path: str):
    """Aggregate data from the latest run's csv folder and save to a JSON file."""
    # Find the latest top-level run folder
    latest_folder = max(glob.glob(os.path.join(base_path, "*")), key=os.path.getmtime)
    if not os.path.isdir(latest_folder):
        logger.error("No valid run folders found!")
        return

    # Initialize JSON structure
    output_data = {}
    # Process each sub-run directory within the latest run folder
    for run_dir in os.listdir(latest_folder):
        run_path = os.path.join(latest_folder, run_dir)
        if os.path.isdir(run_path):
            # Look for the latest folder in the csv subdirectory
            csv_base_path = os.path.join(run_path, "csv")
            if not os.path.isdir(csv_base_path):
                logger.warning(f"No csv directory found in {run_path}. Skipping.")
                continue

            # Find the latest version folder in csv
            latest_csv_folder = max(
                glob.glob(os.path.join(csv_base_path, "version_*")),
                key=os.path.getmtime,
            )
            if not os.path.isdir(latest_csv_folder):
                logger.warning(
                    f"No valid version folder found in {csv_base_path}. Skipping."
                )
                continue

            # Paths to files in the latest csv version folder
            hparams_path = os.path.join(latest_csv_folder, "hparams.yaml")
            metrics_path = os.path.join(latest_csv_folder, "metrics.csv")

            # Check if necessary files exist
            if not os.path.isfile(hparams_path) or not os.path.isfile(metrics_path):
                logger.warning(
                    f"Missing hparams.yaml or metrics.csv in {latest_csv_folder}. Skipping."
                )
                continue

            # Read hparams.yaml
            with open(hparams_path, "r") as file:
                hparams = yaml.safe_load(file)

            # Read metrics.csv and calculate averages
            metrics_df = pd.read_csv(metrics_path)
            avg_train_acc = metrics_df["train_acc"].dropna().mean()
            avg_val_acc = metrics_df["val_acc"].dropna().mean()

            # Create JSON structure for this run
            output_data[f"run_{run_dir}"] = {
                "hparams": hparams,
                "metrics": {"avg_train_acc": avg_train_acc, "avg_val_acc": avg_val_acc},
            }

    # Save aggregated data to JSON
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(
        output_path, f"aggregated_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    logger.info(f"Saving aggregated data to {output_file}")
    with open(output_file, "w") as json_file:
        json.dump(output_data, json_file, indent=4)


if __name__ == "__main__":
    # Paths
    base_path = "./logs/train/runs"
    output_path = "./artifacts"
    multirun_artifact_producer(base_path, output_path)
