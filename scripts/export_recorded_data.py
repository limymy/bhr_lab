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

import signal
import sys
import argparse
import functools
import h5py
import logging
import os
import shutil
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

# Global flag to track interruption
interrupted = False

def signal_handler(signum, frame):
    """Signal handler for graceful interruption."""
    global interrupted
    logging.info("Received interrupt signal. Shutting down gracefully...")
    interrupted = True

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _save_frame_batch(h5_path: str, recorder_path: str, frame_indices: list, frame_ids: list, output_dir: str, compress_level: int):
    """Helper function to save a batch of image frames.
    Args:
        h5_path (str): Path to the HDF5 file.
        recorder_path (str): Path to the recorder group within the HDF5 file.
        frame_indices (list): List of frame indices to process.
        frame_ids (list): List of frame IDs corresponding to the frame indices.
        output_dir (str): The directory to save the PNG files.
        compress_level (int): The compression level for the PNG images.
    """
    # Check if interruption was requested before starting
    if interrupted:
        logging.info(f"    Skipping batch processing due to interruption request")
        return

    try:
        with h5py.File(h5_path, "r") as f:
            recorder_group = f[recorder_path]
            if not isinstance(recorder_group, h5py.Group):
                logging.error(f"    {recorder_path} is not a valid group")
                return

            rgb_data = recorder_group["rgb"]
            if not isinstance(rgb_data, h5py.Dataset):
                logging.error(f"    rgb data in {recorder_path} is not a valid dataset")
                return

            for i, frame_index in enumerate(frame_indices):
                # Check for interruption during processing
                if interrupted:
                    logging.info(f"    Interruption detected, stopping batch processing at frame {frame_index}")
                    return

                try:
                    frame_data = rgb_data[frame_index]
                    frame_id = frame_ids[i]

                    # Ensure frame is in uint8 format for image saving
                    if frame_data.dtype != np.uint8:
                        if isinstance(frame_data, np.ndarray) and np.issubdtype(frame_data.dtype, np.floating):
                            frame_data = (frame_data * 255).astype(np.uint8)
                        else:
                            logging.warning(f"    Frame {frame_index} has an unsupported dtype {frame_data.dtype}, skipping conversion.")
                            continue

                    img = Image.fromarray(frame_data)
                    img.save(os.path.join(output_dir, f"{frame_id}.png"), compress_level=compress_level)
                except Exception as e:
                    logging.error(f"    Failed to save frame {frame_index} (frame_id: {frame_ids[i] if i < len(frame_ids) else 'unknown'}): {e}")
    except Exception as e:
        logging.error(f"    Failed to process batch {frame_indices}: {e}")


def export_camera_data(recorder_group: h5py.Group, output_dir: str, compress_level: int, num_processes: int = None):
    """Exports camera data from a recorder group to a directory of PNG images using multiprocessing.
    Args:
        recorder_group (h5py.Group): The HDF5 group for a specific recorder (e.g., 'cam_left').
        output_dir (str): The directory where PNG files will be saved.
        compress_level (int): The compression level for PNG images (0-9).
        num_processes (int): Number of parallel processes to use. If None, uses os.cpu_count().
    """
    if interrupted:
        logging.info("Camera data export cancelled due to interruption")
        return

    if "rgb" not in recorder_group:
        logging.warning(f"  'rgb' dataset not found in {recorder_group.name}. Skipping.")
        return

    rgb_data = recorder_group["rgb"]
    if not isinstance(rgb_data, h5py.Dataset):
        logging.warning(f"  'rgb' in {recorder_group.name} is not a dataset. Skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)
    num_frames = rgb_data.shape[0]

    # Read timestamp and frame_id data for CSV export
    timestamps = None
    frame_ids = None

    if "timestamp" in recorder_group:
        timestamp_data = recorder_group["timestamp"]
        if isinstance(timestamp_data, h5py.Dataset):
            timestamps = timestamp_data[:]
            logging.info(f"  Found timestamp data with {len(timestamps)} entries")
        else:
            logging.warning(f"  'timestamp' in {recorder_group.name} is not a dataset")

    if "frame_id" in recorder_group:
        frame_id_data = recorder_group["frame_id"]
        if isinstance(frame_id_data, h5py.Dataset):
            frame_ids = frame_id_data[:]
            logging.info(f"  Found frame_id data with {len(frame_ids)} entries")
        else:
            logging.warning(f"  'frame_id' in {recorder_group.name} is not a dataset")

    # Create timestamps.csv if we have the required data
    if timestamps is not None and frame_ids is not None:
        try:
            # Create DataFrame and save to CSV
            df = pd.DataFrame({
                'timestamp': timestamps.flatten() if timestamps.ndim > 1 else timestamps,
                'frame_id': frame_ids.flatten() if frame_ids.ndim > 1 else frame_ids
            })
            csv_path = os.path.join(output_dir, "timestamps.csv")
            df.to_csv(csv_path, index=False)
            logging.info(f"  Successfully saved timestamps.csv to {csv_path}")
        except Exception as e:
            logging.error(f"  Failed to create timestamps.csv: {e}")
            frame_ids = None  # Fall back to index-based naming if CSV creation fails
    else:
        logging.warning(f"  Missing timestamp or frame_id data in {recorder_group.name}, will use index-based naming")

    # Get the HDF5 file path and recorder path for worker processes
    h5_path = rgb_data.file.filename
    recorder_path = recorder_group.name

    logging.info(f"  Saving {num_frames} frames to {output_dir} using multiprocessing...")

    # Check for interruption before starting multiprocessing
    if interrupted:
        logging.info("Multiprocessing cancelled due to interruption")
        return

    # Calculate batch size based on number of CPU cores
    # Determine actual number of processes to use
    if num_processes is None:
        actual_num_processes = os.cpu_count()
    else:
        actual_num_processes = num_processes
    num_processes = os.cpu_count()
    batch_size = max(1, num_frames // actual_num_processes)

    # Create batches of frame indices and corresponding frame_ids
    batches = []
    frame_id_batches = []
    for i in range(0, num_frames, batch_size):
        end_idx = min(i + batch_size, num_frames)
        frame_indices = list(range(i, end_idx))
        batches.append(frame_indices)

        if frame_ids is not None:
            # Use actual frame IDs for naming
            batch_frame_ids = [int(frame_ids[j]) for j in frame_indices]
        else:
            # Fall back to index-based naming
            batch_frame_ids = frame_indices
        frame_id_batches.append(batch_frame_ids)

    # Create a pool of worker processes
    try:
        with Pool(processes=num_processes) as pool:
            # Prepare arguments for each batch
            batch_args = [(h5_path, recorder_path, batch_indices, batch_frame_ids, output_dir, compress_level)
                          for batch_indices, batch_frame_ids in zip(batches, frame_id_batches)]
            # Distribute the batches to the worker processes
            pool.starmap(_save_frame_batch, batch_args)

        logging.info(f"  Finished saving frames from {recorder_group.name}.")
    except KeyboardInterrupt:
        logging.info("  Received interrupt during multiprocessing, cleaning up...")
        # Terminate remaining processes
        pool.terminate()
        pool.join()
        raise


def export_other_data(recorder_group: h5py.Group, output_path: str):
    """Exports non-camera sensor data from a recorder group to a CSV file.

    Args:
        recorder_group (h5py.Group): The HDF5 group for a specific recorder (e.g., 'joint_state').
        output_path (str): The path to the output CSV file.
    """
    if interrupted:
        logging.info("Other data export cancelled due to interruption")
        return

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
    # Set up signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description="Export recorded data from an HDF5 file.")
    parser.add_argument("h5_path", type=str, help="Path to the HDF5 file.")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Directory to save the exported data. Defaults to '<h5_path_dir>/<h5_name>_exported'."
    )
    parser.add_argument(
        "--compress_level", type=int, default=6, help="PNG compression level (0-9)."
    )
    parser.add_argument(
        "--clear_output", type=str, default=None, choices=['yes', 'no'],
        help="Clear output directory if it exists. Options: 'yes', 'no'. If not specified, will prompt user."
    )
    parser.add_argument(
        "--num_processes", type=int, default=None,
        help="Number of parallel processes for camera data export. If not specified, uses all available CPU cores."
    )
    args = parser.parse_args()

    if args.output_dir is None:
        h5_dir = os.path.dirname(args.h5_path)
        h5_name = os.path.splitext(os.path.basename(args.h5_path))[0]
        args.output_dir = os.path.join(h5_dir, f"{h5_name}_exported")

    # Handle output directory clearing
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        if args.clear_output is None:
            # Prompt user for decision
            while True:
                user_input = input(f"Output directory '{args.output_dir}' exists and is not empty. Clear it? (yes/no): ").lower().strip()
                if user_input in ['yes', 'y']:
                    clear_dir = True
                    break
                elif user_input in ['no', 'n']:
                    clear_dir = False
                    break
                else:
                    print("Please enter 'yes' or 'no'.")
        else:
            clear_dir = args.clear_output == 'yes'

        if clear_dir:
            logging.info(f"Clearing output directory: {args.output_dir}")
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir, exist_ok=True)
        else:
            logging.info(f"Keeping existing content in output directory: {args.output_dir}")
    else:
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
                if interrupted:
                    logging.info("Data export cancelled due to interruption")
                    return

                demo_group = data_group.get(demo_name)
                if not isinstance(demo_group, h5py.Group):
                    logging.warning(f"Skipping '{demo_name}' as it is not a group.")
                    continue
                logging.info(f"Processing demo: '{demo_name}'")

                # Create a dedicated output directory for this demo
                demo_output_dir = os.path.join(args.output_dir, demo_name)
                os.makedirs(demo_output_dir, exist_ok=True)
                logging.info(f"  Output directory for this demo: {demo_output_dir}")

                # Iterate over each recorder in the demo (e.g., 'joint_state', 'cam_left')
                for recorder_name in demo_group.keys():
                    if interrupted:
                        logging.info("Recorder processing cancelled due to interruption")
                        return

                    recorder_group = demo_group.get(recorder_name)
                    if not isinstance(recorder_group, h5py.Group):
                        logging.warning(f"  Skipping '{recorder_name}' in '{demo_name}' as it is not a group.")
                        continue
                    logging.info(f"  Processing recorder: '{recorder_name}'")

                    # Check if it's camera data
                    if "cam" in recorder_name.lower():
                        cam_output_dir = os.path.join(demo_output_dir, recorder_name)
                        export_camera_data(recorder_group, cam_output_dir, args.compress_level, args.num_processes)
                    else:
                        # It's other sensor data, export to CSV
                        csv_filename = f"{recorder_name}.csv"
                        csv_output_path = os.path.join(demo_output_dir, csv_filename)
                        export_other_data(recorder_group, csv_output_path)

    except KeyboardInterrupt:
        logging.info("Received interrupt signal during main processing. Exiting gracefully...")
    except FileNotFoundError:
        logging.error(f"Error: The file '{args.h5_path}' was not found.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()