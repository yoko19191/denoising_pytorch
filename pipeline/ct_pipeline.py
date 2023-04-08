""" 
--num_worker 16 --input "../dataset/Kaggle_CT Low Dose Reconstruction/Preprocessed_256x256/256/Full Dose/3mm/Sharp Kernel (D45)" --output "../dataset/Kaggle_CT Low Dose Reconstruction/prepared_recon"
"""

import sys
import os 

import cv2
import numpy as np
import matplotlib.pyplot as plt

import argparse
from tqdm import tqdm

sys.path.insert(0, "..")
import utils.process as process

from concurrent.futures import ThreadPoolExecutor


parser = argparse.ArgumentParser(description="CT Slices Pipeline")

parser.add_argument("--num_worker", help="number of process thread", type=int, required=True, default=4)
parser.add_argument("--input", help="Input prepared sinogram", required=True)
parser.add_argument("--output", help="Output prepared sinogram", required=True)

args = parser.parse_args() 
input_folder = args.input
output_folder = args.output 
num_worker = args.num_worker


def load_ct_slices(folder_path):
    slices = []
    file_names = os.listdir(folder_path)
    file_names.sort()

    for file_name in file_names:
        if file_name.endswith(('.png', '.jpg','.jpeg' ,'.tif', '.bmp')):
            file_path = os.path.join(folder_path, file_name)
            slice_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            slice_image = cv2.resize(slice_image, (256, 256))
            slices.append(slice_image)

    return slices


def get_patient_folders(base_folder_path):
    folder_names = os.listdir(base_folder_path)
    patient_folder_paths = [os.path.join(base_folder_path, folder_name) for folder_name in folder_names if os.path.isdir(os.path.join(base_folder_path, folder_name))]
    return patient_folder_paths


def process_patient_slice(patient_id, patient_slices, noise_level, output_folder, progress_bar):
    noise_output_folder = os.path.join(output_folder, f"lam_{noise_level}", patient_id)
    os.makedirs(noise_output_folder, exist_ok=True)

    for i, slice_image in enumerate(patient_slices):
        projected_slice = process.forward_projection(slice_image)
        noisy_slice = process.add_poisson_noise(projected_slice, noise_level)
        recon_slice = process.filterd_back_projection(noisy_slice)
        
        output_file_path = os.path.join(noise_output_folder, f"noisy_slice_{i:03d}.jpg")
        # cv2.imwrite(output_file_path, noisy_slice)
        cv2.imwrite(output_file_path, recon_slice)
        progress_bar.update(1)


"""load patient's CT slices
"""
patient_folder_paths = get_patient_folders(input_folder)

all_slices = {}

for folder_path in patient_folder_paths:
    patient_id = os.path.basename(folder_path)
    patient_slices = load_ct_slices(folder_path)
    all_slices[patient_id] = patient_slices
    print(f"Loaded {len(patient_slices)} CT slices from {folder_path}")


"""process reconstructed image in different noise_level
"""
noise_levels = [0, 5, 10, 15, 20, 25]

total_slices = sum(len(patient_slices) for patient_slices in all_slices.values())
progress_bar = tqdm(total=total_slices * len(noise_levels), desc="Saving sinograms", ncols=100)

max_workers = num_worker

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    tasks = []
    for noise_level in noise_levels:
        for patient_id, patient_slices in all_slices.items():
            task = executor.submit(process_patient_slice, patient_id, patient_slices, noise_level, output_folder, progress_bar)
            tasks.append(task)

    for task in tasks:
        task.result()

progress_bar.close()


