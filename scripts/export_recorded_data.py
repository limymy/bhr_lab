# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script exports recorded data from an HDF5 file into more accessible formats.

The HDF5 file is expected to have a four-level structure:
data -> demo_* -> recorder_name -> dataset_key

- For non-camera data (e.g., 'joint_state', 'imu'), it combines all datasets for a
  recorder into a single Pandas DataFrame and saves it as a CSV file.
  The filename will be in the format: {demo_name}_{recorder_name}.csv.

- For camera data (identified by 'cam' in the recorder name), it exports the 'rgb'
  dataset into a directory of PNG images. The output path will be:
  {output_dir}/{demo_name}/{recorder_name}/frame_xxxx.png.

- Special handling for 'contact_force' data: The 'forces' dataset, which may have
  a shape of (N, 1, 3), is squeezed to (N, 3) before being saved to CSV.
"""

import argparse
import h5py
import logging
import os
import pandas as pd
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def export_camera_data(recorder_group: h5py.Group, output_dir: str):
    """Exports camera data from a recorder group to a directory of PNG images.

    Args:
        recorder_group (h5py.Group): The HDF5 group for a specific recorder (e.g., 'cam_left').
        output_dir (str): The directory where PNG files will be saved.
    """
    if "rgb" not in recorder_group:
        logging.warning(f"  'rgb' dataset not found in {recorder_group.name}. Skipping.")
        return

    rgb_data = recorder_group["rgb"]
    if not isinstance(rgb_data, h5py.Dataset):
        logging.warning(f"  'rgb' in {recorder_group.name} is not a dataset. Skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"  Saving {rgb_data.shape[0]} frames to {output_dir}...")

    for i in range(rgb_data.shape[0]):
        try:
            frame = rgb_data[i]
            # Ensure frame is in uint8 format for image saving
            if frame.dtype != np.uint8:
                if isinstance(frame, np.ndarray) and np.issubdtype(frame.dtype, np.floating):
                    frame = (frame * 255).astype(np.uint8)
                else:
                    logging.warning(f"    Frame {i} has an unsupported dtype {frame.dtype}, skipping conversion.")
                    continue
            img = Image.fromarray(frame)
            img.save(os.path.join(output_dir, f"frame_{i:04d}.png"))
        except Exception as e:
            logging.error(f"    Failed to save frame {i} from {recorder_group.name}: {e}")


def export_other_data(recorder_group: h5py.Group, output_path: str):
    """Exports non-camera sensor data from a recorder group to a CSV file.

    Args:
        recorder_group (h5py.Group): The HDF5 group for a specific recorder (e.g., 'joint_state').
        output_path (str): The path to the output CSV file.
    """
    data_dict = {}
    logging.info(f"  Processing datasets in {recorder_group.name}...")
    for key, item in recorder_group.items():
        if isinstance(item, h5py.Dataset):
            data = item[:]
            logging.info(f"    Loading dataset '{key}' with original shape {data.shape}")

            # Special handling for contact forces which might be (N, 1, 3)
            if key == "forces" and data.ndim == 3 and data.shape[1] == 1:
                data = np.squeeze(data, axis=1)
                logging.info(f"      Squeezed '{key}' to shape {data.shape}")

            # Flatten multi-dimensional data into multiple columns
            if data.ndim == 1:
                data_dict[key] = data
            elif data.ndim > 1:
                for j in range(data.shape[1]):
                    data_dict[f"{key}_{j}"] = data[:, j]
            else:
                logging.warning(f"    Dataset '{key}' has an unsupported dimension {data.ndim}. Skipping.")

    if not data_dict:
        logging.warning(f"  No datasets found or processed in {recorder_group.name}. Skipping CSV creation.")
        return

    try:
        # Check for consistent lengths before creating DataFrame
        lengths = {key: len(value) for key, value in data_dict.items()}
        if len(set(lengths.values())) > 1:
            logging.error(f"  Inconsistent data lengths in {recorder_group.name}. Cannot create DataFrame.")
            for key, length in lengths.items():
                logging.error(f"    - Dataset '{key}' length: {length}")
            return

        df = pd.DataFrame(data_dict)
        df.to_csv(output_path, index=False)
        logging.info(f"  Successfully saved data to {output_path}")
    except Exception as e:
        logging.error(f"  Failed to create and save DataFrame for {recorder_group.name}: {e}")


def main():
    """Main function to export data from HDF5 file."""
    parser = argparse.ArgumentParser(description="Export recorded data from an HDF5 file.")
    parser.add_argument("h5_path", type=str, help="Path to the HDF5 file.")
    parser.add_argument(
        "--output_dir", type=str, default="output_data", help="Directory to save the exported data."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Output directory: {os.path.abspath(args.output_dir)}")

    try:
        with h5py.File(args.h5_path, "r") as f:
            logging.info(f"Successfully opened HDF5 file: {args.h5_path}")

            if "data" not in f:
                logging.error("HDF5 file does not contain 'data' group at the top level. Aborting.")
                return

            data_group = f["data"]
            demo_names = list(data_group.keys())
            logging.info(f"Found demos: {demo_names}")

            # Iterate over each demo (e.g., 'demo_0', 'demo_1')
            for demo_name in demo_names:
                demo_group = data_group.get(demo_name)
                if not isinstance(demo_group, h5py.Group):
                    logging.warning(f"Skipping '{demo_name}' as it is not a group.")
                    continue
                logging.info(f"Processing demo: '{demo_name}'")

                # Iterate over each recorder in the demo (e.g., 'joint_state', 'cam_left')
                for recorder_name in demo_group.keys():
                    recorder_group = demo_group.get(recorder_name)
                    if not isinstance(recorder_group, h5py.Group):
                        logging.warning(f"  Skipping '{recorder_name}' in '{demo_name}' as it is not a group.")
                        continue
                    logging.info(f"  Processing recorder: '{recorder_name}'")

                    # Check if it's camera data
                    if "cam" in recorder_name.lower():
                        cam_output_dir = os.path.join(args.output_dir, demo_name, recorder_name)
                        export_camera_data(recorder_group, cam_output_dir)
                    else:
                        # It's other sensor data, export to CSV
                        csv_filename = f"{demo_name}_{recorder_name}.csv"
                        csv_output_path = os.path.join(args.output_dir, csv_filename)
                        export_other_data(recorder_group, csv_output_path)

    except FileNotFoundError:
        logging.error(f"Error: The file '{args.h5_path}' was not found.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()