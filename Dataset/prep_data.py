import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


DATASET_DIR = "CVC-ColonDB" 
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
MASKS_DIR = os.path.join(DATASET_DIR, "masks")

# directories for the processed dataset
PROCESSED_DIR = "preprocessed_data1"
TRAIN_IMAGES_DIR = os.path.join(PROCESSED_DIR, "train", "images")
TRAIN_MASKS_DIR = os.path.join(PROCESSED_DIR, "train", "masks")
VALID_IMAGES_DIR = os.path.join(PROCESSED_DIR, "valid", "images")
VALID_MASKS_DIR = os.path.join(PROCESSED_DIR, "valid", "masks")

os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
os.makedirs(TRAIN_MASKS_DIR, exist_ok=True)
os.makedirs(VALID_IMAGES_DIR, exist_ok=True)
os.makedirs(VALID_MASKS_DIR, exist_ok=True)


def load_image(image_path):
    image = cv2.imread(image_path)
    return image


def create_and_write_image_mask(image_paths, mask_paths, save_images_dir, save_masks_dir):
    for image_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths), desc="Processing images and masks"):
        image = load_image(image_path)
        mask = load_image(mask_path)

        
        image_filename = os.path.splitext(os.path.basename(image_path))[0] + ".png"
        mask_filename = os.path.splitext(os.path.basename(mask_path))[0] + ".png"

        cv2.imwrite(os.path.join(save_images_dir, image_filename), image)
        cv2.imwrite(os.path.join(save_masks_dir, mask_filename), mask)


image_files = sorted([os.path.join(IMAGES_DIR, filename) for filename in os.listdir(IMAGES_DIR)])
mask_files = sorted([os.path.join(MASKS_DIR, filename) for filename in os.listdir(MASKS_DIR)])

# Split dataset into train and valid sets
train_images, valid_images, train_masks, valid_masks = train_test_split(image_files, mask_files, train_size=0.8, shuffle=True, random_state=42)


create_and_write_image_mask(train_images, train_masks, TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR)
create_and_write_image_mask(valid_images, valid_masks, VALID_IMAGES_DIR, VALID_MASKS_DIR)
