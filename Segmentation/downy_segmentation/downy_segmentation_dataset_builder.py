"""downy_segmentation dataset."""

import tensorflow_datasets as tfds
import os
import json
import cv2
import numpy as np
import tensorflow as tf
import random
# from keras_unet.utils import get_patches  # !pip install keras-unet


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

def get_img_and_mask(json_file, data_dir):

    with open(json_file, "r") as f:
        data = json.load(f)

    image_path = extract_image_path(json_file, data_dir)
    img = read_image(image_path)
    
    mask_size = img.shape

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

    mask = np.zeros(mask_size, np.uint8)

    for mask_class, vertices in all_polygons:
        if mask_class == 1:
            mask = cv2.fillPoly(mask, pts=[vertices], color=(255, 0, 0))

    size = (2048, 2048)  # Version 6.0.0

    img = cv2.resize(img, size)    
    mask = cv2.resize(mask, size)

    return img, mask[..., :1]


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for downy_segmentation dataset."""

    VERSION = tfds.core.Version("6.0.0")
    RELEASE_NOTES = {
        "2.0.0": "(512, 512)",
        "3.0.0": "shuffled",
        "4.0.0": "(1024, 1024)",
        "5.0.0": "(256, 256)",
        "6.0.0": "(2048, 2048)"
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

        all_labels = sorted(
            [
                f"{self.extracted_labels_dir}/{p}"
                for p in os.listdir(self.extracted_labels_dir)
                if "JPG" in p
            ]
        )

        random.seed(42)
        random.shuffle(all_labels)

        train_labels, test_labels = all_labels[:379], all_labels[379:]  # (379, 43)
        # train_labels, test_labels = all_labels[:10], all_labels[-10:]  # (379, 43)

        return {
            "test": self._generate_examples(test_labels),
            "train": self._generate_examples(train_labels),
        }

    def _generate_examples(self, labels):
        """Yields examples."""
        
        for i, label in enumerate(labels):
            try:
                img, mask = get_img_and_mask(label, self.dataset_dir)
                yield i, {
                    "image": img,
                    "mask": mask,
                }
            except:
                print(label)

