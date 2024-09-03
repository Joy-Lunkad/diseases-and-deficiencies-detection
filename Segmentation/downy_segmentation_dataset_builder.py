"""downy_segmentation dataset."""
from __future__ import annotations

import tensorflow_datasets as tfds
import os
import json
import cv2
import numpy as np
import random


def extract_image_path(json_path, image_dir):
    """
    Given a string representing the path to a JSON file of the
    form "extracted_labels/_DSC2179.JPG.json",
    returns only the image path "_DSC2179.JPG".
    """
    return image_dir + "/" + os.path.basename(json_path)[:-5]


def read_image(image_path):
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # im = cv2.resize(im, (3712, 5568))  # Version 0.0.0
    # im = cv2.resize(im, (2048, 2048))  # Version 1.0.0
    return im

def patchify(image, mask, patch_size):
    """Patches an image and mask into smaller pieces of size patch_size x patch_size"""
    
    all = np.concatenate([image, mask,], axis=-1)
    
    H, W, C = all.shape
    all = all.reshape(H // patch_size, patch_size, W//patch_size, patch_size, C)
    all = all.swapaxes(1, 2)
    patches = all.reshape(-1, patch_size, patch_size, C)

    image = patches[..., :3]
    mask = patches[..., 3:4]

    # which_masks = np.sum(mask, axis=(1, 2, 3))
    # keep = np.where(which_masks > 0)[0]
    
    # Only for version 8
    # without_downy = np.where(which_masks == 0)[0]
    # if len(without_downy) <= len(keep):
    #     num_wo_to_keep = len(without_downy)
    # else:
    #     num_wo_to_keep = len(keep)

    # if num_wo_to_keep:
    #     keep_wo = random.sample(without_downy.tolist(), num_wo_to_keep)
    #     keep = np.concatenate([keep, keep_wo])

    # mask = mask[keep]
    # image = image[keep]
    
    return image, mask


def get_img_and_mask(json_file, data_dir, resize_rez, patch_size: int | None = None):
    
    image_path = extract_image_path(json_file, data_dir)
    img = read_image(image_path)

    with open(json_file, "r") as f:
        data = json.load(f)

    all_segmented_masks = data["instances"]
    all_polygons = []

    for mask in all_segmented_masks:
        mask_class = mask["classId"]
        points = mask["points"]

        new_points = []
        for i in range(0, len(points), 2):
            new_points.append((points[i], points[i + 1]))

        vertices = np.ceil(np.array(new_points)).astype(np.int32)
        all_polygons.append([mask_class, vertices])

    mask = np.zeros(img.shape, np.uint8)

    for mask_class, vertices in all_polygons:
        if mask_class == 1:
            mask = cv2.fillPoly(mask, pts=[vertices], color=(255, 0, 0))

    img = cv2.resize(img, resize_rez)
    mask = cv2.resize(mask, resize_rez)
    mask = mask[..., :1]

    if patch_size:
        images, masks = patchify(img, mask, patch_size)
    else:
        images = np.expand_dims(img, axis=0)
        masks = np.expand_dims(mask, axis=0)

    return images, masks


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for downy_segmentation dataset."""

    VERSION = tfds.core.Version("9.1024.256")
    RELEASE_NOTES = {
        "2.0.0": "(512, 512)",
        "3.0.0": "shuffled",
        "4.0.0": "(1024, 1024)",
        "5.0.0": "(256, 256)",
        "6.0.0": "(2048, 2048)",
        
        "7.256.0": "(256, 256), 2000+ images total, both from 60 cm and 90 cm",
        "7.256.60": "(256, 256), 1000+ images total, from 60 cm",
        "7.256.90": "(256, 256), 1000+ images total, from 90 cm",
        
        "7.512.0": "(512, 512), 2000+ images total, both from 60 cm and 90 cm",
        "7.512.60": "(512, 512), 1000+ images total, from 60 cm",
        "7.512.90": "(512, 512), 1000+ images total, from 90 cm",
        "7.512.256": "(512, 512), 2000+ images total"
                    "but patchified to (256, 256), patches without any downy mildew are removed",
        "8.512.256": "(512, 512), 2000+ images total"
                    "but patchified to (256, 256), some patches without any downy mildew are kept",
        "9.512.256": "(512, 512), 2000+ images total, but patchified to (256, 256)",
    
        "7.1024.0": "(1024, 1024), 2000+ images total, both from 60 cm and 90 cm",
        "7.1024.60": "(1024, 1024), 1000+ images total, from 60 cm",
        "7.1024.90": "(1024, 1024), 1000+ images total, from 90 cm",
        "7.1024.256": "(1024, 1024), 2000+ images total"
                    "but patchified to (256, 256), patches without any downy mildew are removed",
        "8.1024.256": "(1024, 1024), 2000+ images total"
                    "but patchified to (256, 256), some patches without any downy mildew are kept",
        "9.1024.256": "(1024, 1024), 2000+ images total, but patchified to (256, 256)",
        
        
        "7.2048.0": "(2048, 2048), 2000+ images total, both from 60 cm and 90 cm",
        "7.2048.60": "(2048, 2048), 1000+ images total, from 60 cm",
        "7.2048.90": "(2048, 2048), 1000+ images total, from 90 cm",
        "7.2048.256": "(2048, 2048), 2000+ images total, but patchified to (256, 256), also patches without any downy mildew are removed",
        "8.2048.256": "(2048, 2048), 2000+ images total, but patchified to (256, 256), but some patches without any downy mildew are kept",
        
        "7.3712.0": "(3712, 5568), 2000+ images total, both from 60 cm and 90 cm",
        "7.3712.60": "(3712, 5568), 1000+ images total, from 60 cm",
        "7.3712.90": "(3712, 5568), 1000+ images total, from 90 cm",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(downy_segmentation): Specifies the tfds.core.DatasetInfo object

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "image": tfds.features.Image(shape=(None, None, 3)),
                    "mask": tfds.features.Image(shape=(None, None, 1)),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("image", "mask"),  # Set to `None` to disable
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(downy_segmentation): Downloads the data and defines the splits

        self.base_path = "/content/drive/MyDrive/ICAR ML/Dataset/Segmentation"
        self.dataset_dir = self.base_path + "/Downy mildew distance images"
        self.extracted_labels_dir = self.base_path + "/extracted_labels"

        self.sixty_labels = sorted(
            [
                f"{self.extracted_labels_dir}_60_cm/{p}"
                for p in os.listdir(self.extracted_labels_dir + "_60_cm")
                if "JPG" in p
            ]
        )
        
        self.ninety_labels = sorted(
            [
                f"{self.extracted_labels_dir}_90_cm/{p}"
                for p in os.listdir(self.extracted_labels_dir + "_90_cm")
                if "JPG" in p
            ]
        )
        
        self.resize_rez = self.VERSION.minor
        which_distance_or_patch_size = self.VERSION.patch
        self.patch_size = None

        if which_distance_or_patch_size == 60:
            self.all_labels = self.sixty_labels
        elif which_distance_or_patch_size == 90:
            self.all_labels = self.ninety_labels
        else:
            self.all_labels = self.sixty_labels + self.ninety_labels
            self.patch_size = which_distance_or_patch_size
            
        random.seed(42)
        random.shuffle(self.all_labels)
        
        total_labels = len(self.all_labels)
        split = int(0.9 * total_labels)
        train_labels, test_labels = self.all_labels[:split], self.all_labels[split:]

        return {
            "test": self._generate_examples(test_labels),
            "train": self._generate_examples(train_labels),
        }

    def _generate_examples(self, labels):
        """Yields examples."""

        print(f"resize_resolution: {self.resize_rez}")

        counter = 0
        for label in labels:
            try:
                images, masks = get_img_and_mask(
                    label,
                    self.dataset_dir,
                    (self.resize_rez, self.resize_rez),
                    self.patch_size,
                )

                for img, mask in zip(images, masks):
                    yield counter, {
                        "image": img,
                        "mask": mask,
                    }
                    counter += 1
                    
            except:
                print(label)
