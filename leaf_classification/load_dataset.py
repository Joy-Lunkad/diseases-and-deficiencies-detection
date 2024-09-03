from __future__ import annotations
import re
from typing import Any
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.applications import imagenet_utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import functools
import albumentations as A
import tensorflow_probability as tfp

tfd = tfp.distributions

RESOLUTIONS = {"5.0.0": "256", "3.0.0": "512", "4.0.0": "1024", "6.0.0": "2048"}
TOTAL_TRAIN_IMAGES = 377
TOTAL_TEST_IMAGES = 42


def build_aug_transform():
    # TODO: add types for light, medium, heavy augs.

    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            # Spacial Augs
            A.OneOf(
                [
                    A.ElasticTransform(
                        p=0.5, alpha=1, sigma=30, alpha_affine=40, approximate=True
                    ),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
                ],
                p=0.5,
            ),
            # Colour Aug
            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=0.5),
                    A.RandomGamma(p=0.8),
                    A.CLAHE(p=0.8),
                    A.HueSaturationValue(
                        p=0.1
                    ),  # Changing color scheme might be harmful.
                ],
                p=0.8,
            ),
            # Power Regularizing Augs
            A.OneOf(
                [
                    A.RandomGridShuffle(p=0.5),
                    A.PixelDropout(p=0.1, mask_drop_value=None),
                    A.GridDropout(p=0.1, ratio=0.25),
                ],
                p=0.2,
            ),
            # Quality Augs
            A.OneOf(
                [
                    # A.Blur(blur_limit=3, p=0.2),
                    A.Sharpen(p=0.5),
                    A.GaussNoise(p=0.5),
                ],
                p=0.5,
            ),
            # Weather Augs
            A.RandomShadow(p=0.2),
        ]
    )
    return transform


def random_crop(data, crop_h, crop_w):
    """Data should not be batched"""
    both = tf.concat([data["image"], data["mask"]], axis=-1)
    both = tf.image.random_crop(both, size=(crop_h, crop_w, 4))
    return {"image": both[..., :3], "mask": both[..., -1:]}


def patchify(data, patch_size):
    """Data should be batched"""

    image, mask, sample_weights = patch(
        image=data["image"],
        mask=data["mask"],
        sample_weights=data["sample_weights"],
        patch_size=patch_size,
    )

    return {
        "image": image,
        "mask": mask,
        "sample_weights": sample_weights,
    }


def augment(data, transform):
    """Data should not be batched"""

    def _augment_image(image, mask):
        """Data should not be batched"""
        augmented = transform(image=image.numpy(), mask=mask.numpy())
        image = tf.convert_to_tensor(augmented["image"], dtype=tf.float32)
        mask = tf.convert_to_tensor(augmented["mask"], dtype=tf.float32)
        return image, mask

    image, mask = tf.py_function(
        func=_augment_image,
        inp=[data["image"], data["mask"]],
        Tout=[tf.float32, tf.float32],
    )
    image.set_shape(data["image"].shape)
    mask.set_shape(data["mask"].shape)

    return {"image": image, "mask": mask}


def _get_sample_weights(fg_sample_count, bg_sample_count: int) -> tuple:
    total_sample_count = fg_sample_count + bg_sample_count

    fg_weight = total_sample_count // (2 * fg_sample_count)
    bg_weight = total_sample_count // (2 * bg_sample_count)

    return fg_weight, bg_weight


def _normalize(data: dict[str, Any], bg_weight, fg_weight):
    """Data should not be batched"""
    image = imagenet_utils.preprocess_input(
        tf.cast(data["image"], tf.float32), mode="tf"
    )

    # image = tf.cast(data["image"], tf.float32) / 255.0  # type: ignore
    mask = tf.cast(data["mask"], tf.float32) / 255.0  # type: ignore

    # print("MASK type", mask, type(data['mask']))

    fg_sample_count = tf.reduce_sum(mask)
    # fg_sample_count = np.sum(data['mask'])

    total_sample_count = tf.cast(tf.size(mask), tf.float32)
    # total_sample_count = tf.cast(tf.size(mask), tf.float32)

    # print("Datatype of sc", total_sample_count, fg_sample_count)

    # total_sample_count = float(total_sample_count)

    bg_sample_count = total_sample_count - fg_sample_count

    # print(
    #     "fg_sample_count {} total_sample_count {} bg_sample_count {}".format(
    #         fg_sample_count, total_sample_count, bg_sample_count
    #     )
    # )

    fg_sample_weight, bg_sample_weight = _get_sample_weights(fg_sample_count, bg_sample_count)

    # sample_weights = tf.where(
    #     mask == 0,
    #     bg_weight = bg_sample_weight,
    #     fg_weight = fg_sample_weight,
    # )

    sample_weights = tf.where(
        mask == 0,
        bg_sample_weight,
        fg_sample_weight,
    )

    # print(" bg weight and fg weight", bg_weight, fg_weight)
    return {
        "image": image,
        "mask": mask,
        "sample_weights": sample_weights,
    }


def cutmix_padding(h, w):
    """Returns image mask for CutMix.

    Taken from (https://github.com/google/edward2/blob/master/experimental
    /marginalization_mixup/data_utils.py#L367)

    Args:
      h: image height.
      w: image width.
    """
    r_x = tf.random.uniform([], 0, w, tf.int32)
    r_y = tf.random.uniform([], 0, h, tf.int32)

    # Beta dist in paper, but they used Beta(1,1) which is just uniform.
    image1_proportion = tf.random.uniform([])
    patch_length_ratio = tf.math.sqrt(1 - image1_proportion)
    r_w: Any = tf.cast(patch_length_ratio * tf.cast(w, tf.float32), tf.int32)
    r_h: Any = tf.cast(patch_length_ratio * tf.cast(h, tf.float32), tf.int32)
    bbx1: Any = tf.clip_by_value(tf.cast(r_x - r_w // 2, tf.int32), 0, w)
    bby1: Any = tf.clip_by_value(tf.cast(r_y - r_h // 2, tf.int32), 0, h)
    bbx2: Any = tf.clip_by_value(tf.cast(r_x + r_w // 2, tf.int32), 0, w)
    bby2: Any = tf.clip_by_value(tf.cast(r_y + r_h // 2, tf.int32), 0, h)

    # Create the binary mask.
    pad_left = bbx1
    pad_top = bby1
    pad_right = tf.maximum(w - bbx2, 0)
    pad_bottom = tf.maximum(h - bby2, 0)
    r_h = bby2 - bby1
    r_w = bbx2 - bbx1

    mask = tf.pad(
        tf.ones((r_h, r_w)),
        paddings=[[pad_top, pad_bottom], [pad_left, pad_right]],
        mode="CONSTANT",
        constant_values=0,
    )
    mask.set_shape((h, w))
    return mask[..., None]  # Add channel dim.


def _preprocess_for_cutmix_or_mixup(sample, h, w, augment_name):
    if augment_name is not None and "cutmix" in augment_name:
        sample["cutmix_mask"] = cutmix_padding(h, w)
        sample["cutmix_ratio"] = tf.reduce_mean(sample["cutmix_mask"])

    if augment_name is not None and "mixup" in augment_name:
        mixup_alpha = 0.2  # default to alpha=0.2
        # If float provided, get it
        if "mixup_" in augment_name:
            alpha = augment_name.split("mixup_")[1].split("_")
            if any(alpha) and re.match(r"^-?\d+(?:\.\d+)?$", alpha[0]) is not None:
                mixup_alpha = float(alpha[0])
        beta = tfd.Beta(mixup_alpha, mixup_alpha)
        sample["mixup_ratio"] = beta.sample()

    return sample


def _apply_cutmix(batch):
    batch = dict(**batch)
    bs = tf.shape(batch["image"])[0] // 2
    mask = batch["cutmix_mask"][:bs]
    images = mask * batch["image"][:bs] + (1.0 - mask) * batch["image"][bs:]
    masks = mask * batch["mask"][:bs] + (1.0 - mask) * batch["mask"][bs:]
    sample_weights = (
        mask * batch["sample_weights"][:bs]
        + (1.0 - mask) * batch["sample_weights"][bs:]
    )

    out_batch = {
        "image": images,
        "mask": masks,
        "sample_weights": sample_weights,
    }
    return out_batch


def _apply_mixup(batch):
    """Mixup."""
    batch = dict(**batch)
    bs = tf.shape(batch["image"])[0] // 2
    ratio = batch["mixup_ratio"][:bs, None, None, None]
    images = ratio * batch["image"][:bs] + (1.0 - ratio) * batch["image"][bs:]
    masks = ratio * batch["mask"][:bs] + (1.0 - ratio) * batch["mask"][bs:]
    sample_weights = (
        ratio * batch["sample_weights"][:bs]
        + (1.0 - ratio) * batch["sample_weights"][bs:]
    )

    out_batch = {
        "image": images,
        "mask": masks,
        "sample_weights": sample_weights,
    }

    return out_batch


def _apply_either_mixup_or_cutmix(batch):
    """Randomly applies one of cutmix or mixup to a batch."""
    return tf.cond(
        tf.cast(tf.random.uniform([], maxval=2, dtype=tf.int32), tf.bool),
        lambda: _apply_mixup(batch),
        lambda: _apply_cutmix(batch),
    )


def _apply_both_mixup_cutmix(batch):
    """Apply mixup to half the batch, and cutmix to the other."""
    batch = dict(**batch)
    bs = tf.shape(batch["image"])[0] // 4

    mixup_ratio = batch["mixup_ratio"][:bs, None, None, None]
    mixup_images = (
        mixup_ratio * batch["image"][:bs]
        + (1.0 - mixup_ratio) * batch["image"][bs : 2 * bs]
    )

    # mixup_masks = (mixup_ratio * batch['mask'][:bs]
    #                 + (1.0 - mixup_ratio) * batch['mask'][bs:2*bs])
    # mixup_weights = (mixup_ratio * batch['sample_weights'][:bs]
    #                 + (1.0 - mixup_ratio) * batch['sample_weights'][bs:2*bs])

    mixup_masks = batch["mask"][:bs] + batch["mask"][bs : 2 * bs]
    mixup_weights = batch["sample_weights"][:bs] + batch["sample_weights"][bs : 2 * bs]

    cutmix_mask = batch["cutmix_mask"][2 * bs : 3 * bs]
    cutmix_images = (
        cutmix_mask * batch["image"][2 * bs : 3 * bs]
        + (1.0 - cutmix_mask) * batch["image"][-bs:]
    )

    cutmix_masks = (
        cutmix_mask * batch["mask"][2 * bs : 3 * bs]
        + (1.0 - cutmix_mask) * batch["mask"][-bs:]
    )
    cutmix_weights = (
        cutmix_mask * batch["sample_weights"][2 * bs : 3 * bs]
        + (1.0 - cutmix_mask) * batch["sample_weights"][-bs:]
    )

    return {
        "image": tf.concat([mixup_images, cutmix_images], axis=0),
        "mask": tf.concat([mixup_masks, cutmix_masks], axis=0),
        "sample_weights": tf.concat([mixup_weights, cutmix_weights], axis=0),
    }


@tf.function
def preprocess_ds(
    ds: tf.data.Dataset,
    augment_name: str | None,
    is_training: bool,
    batch_size: int,
    foreground_weight: float,
    background_weight: float,
    image_shape: tuple[int, int],
):
    h, w = image_shape

    if augment_name is None:
        augment_name = ""

    normalize = functools.partial(
        _normalize,
        fg_weight=foreground_weight,
        bg_weight=background_weight,
    )

    ds = ds.repeat()

    if is_training:
        if "randcrop_" in augment_name:
            crop_size = int(augment_name.split("randcrop_")[1].split("_")[0])
            crop_fn = functools.partial(random_crop, crop_h=crop_size, crop_w=crop_size)
            ds = ds.map(crop_fn)
            h, w = crop_size, crop_size

        if "transform" in augment_name:
            aug_fn = functools.partial(augment, transform=build_aug_transform())
            ds = ds.map(aug_fn)

        ds = ds.map(normalize)

        if "cutmix" in augment_name or "mixup" in augment_name:
            preprocess_for_cutmix_or_mixup = functools.partial(
                _preprocess_for_cutmix_or_mixup,
                h=h,  # type: ignore
                w=w,  # type: ignore
                augment_name=augment_name,
            )
            ds = ds.map(preprocess_for_cutmix_or_mixup)
            ds = ds.batch(batch_size * 2, drop_remainder=True)

            if "cutmix" in augment_name and "mixup" not in augment_name:
                ds = ds.map(_apply_cutmix)

            elif "mixup" in augment_name and "cutmix" not in augment_name:
                ds = ds.map(_apply_mixup)

            elif "cutmix" in augment_name and "mixup" in augment_name:
                if batch_size > 1:
                    ds = ds.map(_apply_both_mixup_cutmix)
                else:
                    ds = ds.map(_apply_either_mixup_or_cutmix)
        else:
            ds = ds.batch(batch_size, drop_remainder=True)

    else:
        ds = ds.map(normalize).batch(batch_size, drop_remainder=True)

    if "patchify_" in augment_name:
        patch_size = int(augment_name.split("patchify_")[1].split("_")[0])
        patchify_fn = functools.partial(patchify, patch_size=patch_size)
        ds = ds.map(patchify_fn)

    return ds


def prepare_ds(
    version,
    batch_size,
    augment_name="",
    foreground_weight=1e3,
    background_weight=0.0,
    use_sample_weights=True,
    data_dir="./",
):
    h = w = int(RESOLUTIONS[version])

    ds_train: tf.data.Dataset = tfds.load(
        f"downy_segmentation:{version}",
        split="train",
        data_dir=data_dir,
        download=False,
    )  # type: ignore

    ds_test: tf.data.Dataset = tfds.load(
        f"downy_segmentation:{version}",
        split="test",
        data_dir=data_dir,
        download=False,
    )  # type: ignore

    ds_train = ds_train.shuffle(batch_size * 10, reshuffle_each_iteration=True)
    ds_test = ds_test.shuffle(batch_size * 10, reshuffle_each_iteration=True)

    ds_train = preprocess_ds(
        ds_train,
        augment_name=augment_name,
        is_training=True,
        batch_size=batch_size,
        foreground_weight=foreground_weight,
        background_weight=background_weight,
        image_shape=(h, w),
    )  # type: ignore

    test_aug = ""
    if "patchify_" in augment_name:
        patch_size = int(augment_name.split("patchify_")[1].split("_")[0])
        test_aug = f"patchify_{patch_size}"

    elif "randcrop_" in augment_name:
        crop_size = int(augment_name.split("randcrop_")[1].split("_")[0])
        test_aug = f"patchify_{crop_size}"

    ds_test = preprocess_ds(
        ds_test,
        augment_name=test_aug,
        is_training=False,
        batch_size=batch_size,
        foreground_weight=foreground_weight,
        background_weight=background_weight,
        image_shape=(h, w),
    )  # type: ignore

    if "patchify_" in test_aug:
        ds_test = ds_test.unbatch().batch(batch_size, drop_remainder=True)

    def unpack(batch):
        if use_sample_weights:
            return batch["image"], batch["mask"], batch["sample_weights"]
        else:
            return batch["image"], batch["mask"]

    ds_train = ds_train.map(unpack).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.map(unpack).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test


def patch(image, mask, sample_weights, patch_size):
    """To be run after batching"""

    all: Any = tf.concat([image, mask, sample_weights], axis=-1)
    patches = tf.image.extract_patches(
        all,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )

    patches = tf.reshape(patches, (-1, patch_size, patch_size, all.shape[-1]))

    im_patches = patches[..., :3]
    m_patches = patches[..., 3:4]
    sw_patches = patches[..., 4:]

    return im_patches, m_patches, sw_patches


def join_patches(x: Any, batch_size: int, patch_size: int) -> np.ndarray:
    b, h, w, c = x.shape
    x = tf.reshape(x, (batch_size, -1, *x.shape[-3:]))
    grid_shape = int(x.shape[1] ** 0.5)
    x = tf.reshape(x, (batch_size, grid_shape, grid_shape, *x.shape[-3:])).numpy()
    placeholder = np.zeros(shape=(batch_size, patch_size, patch_size, c), dtype=x.dtype)

    for b in range(batch_size):
        for i in range(grid_shape):
            for j in range(grid_shape):
                x1, x2 = i * patch_size, (i + 1) * patch_size
                y1, y2 = j * patch_size, (j + 1) * patch_size
                placeholder[b][x1:x2, y1:y2] = x[b][i, j]

    return placeholder


def plot_patches(patches: Any, batch_size: int):
    """Only plots the first sample in the batch"""

    patches = tf.reshape(patches, (batch_size, -1, *patches.shape[-3:]))
    grid_shape = int(patches.shape[1] ** 0.5)

    patches = tf.reshape(
        patches, (batch_size, grid_shape, grid_shape, *patches.shape[-3:])
    ).numpy()

    fig, axes = plt.subplots(grid_shape, grid_shape, figsize=(5, 5))

    for i in range(grid_shape):
        for j in range(grid_shape):
            patch = patches[0][i, j]
            axes[i, j].imshow(patch)
            axes[i, j].axis("off")

    fig.tight_layout(pad=0)
    plt.show()
