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
import enum

tfd = tfp.distributions

RESOLUTIONS = {
    "5.0.0": "256",
    "3.0.0": "512",
    "4.0.0": "1024",
    "6.0.0": "2048",
    "7.0.0": "256",
    "7.512.256": "256",
    "8.512.256": "256",
    "9.512.256": "256",
    "7.2048.256": "256",
    "8.2048.256": "256",
}

SPECIFIC_TOTAL_TRAIN_IMAGES = {
    "9.512.256": 7780,
    # "7.512.256": 11090, TODO
    # "7.2048.256": 11090,
    # "8.512.256": 22180, TODO
    # "8.2048.256": 22180,
}

SPECIFIC_TOTAL_TEST_IMAGES = {
    "9.512.256": 864,
    # "7.512.256": 1360, TODO
    # "7.2048.256": 1360,
    # "8.512.256": 2720, TODO
    # "8.2048.256": 2720,
}


def get_total_train_images(version) -> int:
    return SPECIFIC_TOTAL_TRAIN_IMAGES.get(version, 1945)


def get_total_test_images(version) -> int:
    return SPECIFIC_TOTAL_TEST_IMAGES.get(version, 216)


def get_resolution(version) -> int:
    return int(RESOLUTIONS.get(version, tfds.core.Version(version).minor))  # type: ignore


def get_patch_size(version) -> int:
    return int(tfds.core.Version(version).patch)  # type: ignore


def get_prepatchify_resolution(version) -> int:
    return int(tfds.core.Version(version).minor)  # type: ignore


class AugType(enum.Enum):
    """Imagenet dataset split."""

    LIGHT = 1
    MED = 2
    HEAVY = 3
    MAX = 4

    @classmethod
    def from_string(cls, name: str) -> "AugType":
        return {
            "LIGHT": AugType.LIGHT,
            "MED": AugType.MED,
            "HEAVY": AugType.HEAVY,
            "MAX": AugType.MAX,
        }[name.upper()]

    @property
    def spacial_p(self):
        return {
            AugType.LIGHT: 0.0,
            AugType.MED: 0.1,
            AugType.HEAVY: 0.3,
            AugType.MAX: 0.5,
        }[self]

    @property
    def color_p(self):
        return {
            AugType.LIGHT: 0.1,
            AugType.MED: 0.3,
            AugType.HEAVY: 0.5,
            AugType.MAX: 0.8,
        }[self]

    @property
    def regularizing_p(self):
        return {
            AugType.LIGHT: 0.0,
            AugType.MED: 0.1,
            AugType.HEAVY: 0.2,
            AugType.MAX: 0.3,
        }[self]

    @property
    def gridshuffle_p(self):
        return {
            AugType.LIGHT: 0.3,
            AugType.MED: 0.3,
            AugType.HEAVY: 0.5,
            AugType.MAX: 0.7,
        }[self]

    @property
    def quality_p(self):
        return {
            AugType.LIGHT: 0.1,
            AugType.MED: 0.2,
            AugType.HEAVY: 0.3,
            AugType.MAX: 0.5,
        }[self]

    @property
    def weather_p(self):
        return {
            AugType.LIGHT: 0.0,
            AugType.MED: 0.0,
            AugType.HEAVY: 0.1,
            AugType.MAX: 0.2,
        }[self]


def build_aug_transform(augtype: AugType):
    print(f"Using AugType: {augtype}")

    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            # Spacial Augs
            A.OneOf(
                [
                    A.ElasticTransform(
                        p=0.5, alpha=1, sigma=30, alpha_affine=30, approximate=True
                    ),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
                ],
                p=augtype.spacial_p,
            ),
            # Colour Aug
            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=0.5),
                    A.RandomGamma(p=0.8),
                    A.CLAHE(p=0.8),
                    # A.HueSaturationValue(
                    #     p=0.1
                    # ),  # Changing color scheme might be harmful.
                ],
                p=augtype.color_p,
            ),
            # Power Regularizing Augs
            A.OneOf(
                [
                    A.PixelDropout(p=0.1, mask_drop_value=None),
                    A.GridDropout(p=0.1, ratio=0.25),
                ],
                p=augtype.regularizing_p,
            ),
            A.RandomGridShuffle(p=augtype.gridshuffle_p),
            # Quality Augs
            A.OneOf(
                [
                    # A.Blur(blur_limit=3, p=0.2),
                    A.Sharpen(p=0.5),
                    A.GaussNoise(p=0.5),
                ],
                p=augtype.quality_p,
            ),
            # Weather Augs
            A.RandomShadow(p=augtype.weather_p),
        ]
    )
    return transform


def random_crop(data, crop_h, crop_w):
    """Data should not be batched"""
    both = tf.concat([data["image"], data["mask"]], axis=-1)
    both = tf.image.random_crop(both, size=(crop_h, crop_w, 4))
    return {"image": both[..., :3], "mask": both[..., -1:]}


def patchify(data, patch_size, reduce_batch_by_factor):
    """Data should be batched"""

    image, mask = patch(image=data["image"], mask=data["mask"], patch_size=patch_size)

    out = {"image": image, "mask": mask}
        
    if reduce_batch_by_factor > 1: 
        out = reduce_batch_by_remove_samples_with_less_downy(
            data=out,
            prepatchify_batch_size=tf.shape(data["image"])[0],
            prepatchify_im_size=tf.shape(data["image"])[1],
            patch_size=patch_size,
            reduce_batch_by_factor=reduce_batch_by_factor,
        )

    return out


def reduce_batch_by_remove_samples_with_less_downy(
    data,
    prepatchify_batch_size,
    prepatchify_im_size,
    patch_size,
    reduce_batch_by_factor,
):
    
    print(f"{prepatchify_batch_size=}")
    print(f"{prepatchify_im_size=}")
    print(f"{patch_size=}")
    print(f"{reduce_batch_by_factor=}")
    
    image, mask = data["image"], data["mask"]
    if reduce_batch_by_factor > 1:
        num_patches_per_sample = (prepatchify_im_size // patch_size) ** 2
        num_patches_to_keep = prepatchify_batch_size * (
            num_patches_per_sample // reduce_batch_by_factor
        )

        which_masks = tf.reduce_sum(mask, axis=(1, 2, 3))
        sort_order = tf.argsort(which_masks)
        keep = tf.random.shuffle(sort_order[-num_patches_to_keep:])
        mask = tf.gather(mask, keep)
        image = tf.gather(image, keep)

    return {"image": image, "mask": mask}


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


def _get_dynamic_sample_weights(mask, fg_w):
    """
    Returns dymanic sample weights for the background and foreground classes.
    Formula forces fg_w * no_of_fg_pixels = bg_w * no_of_bg_pixels when
    fg_w = 1. but allows for fg_w to be changed in a way to scale up the
    relative importance of fg_pixels. If fg_w = 10, each fg_pixel is 10x
    valuable as a bg_pixel. fg_w is a hyperparameter to be tuned.
    """
    n_fg = tf.reduce_sum(mask)
    n_bg = tf.cast(tf.size(mask), tf.float32) - n_fg
    bg_w = n_fg / n_bg
    return (bg_w, fg_w)


def _normalize(data: dict[str, Any]):
    """Data should not be batched"""
    image = imagenet_utils.preprocess_input(
        tf.cast(data["image"], tf.float32), mode="tf"
    )

    mask = tf.cast(data["mask"], tf.float32) / 255.0  # type: ignore

    return {
        "image": image,
        "mask": mask,
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
    if augment_name and "cutmix" in augment_name:
        sample["cutmix_mask"] = cutmix_padding(h, w)
        sample["cutmix_ratio"] = tf.reduce_mean(sample["cutmix_mask"])

    if augment_name and "mixup" in augment_name:
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
    # sample_weights = (
    #     mask * batch["sample_weights"][:bs]
    #     + (1.0 - mask) * batch["sample_weights"][bs:]
    # )

    out_batch = {
        "image": images,
        "mask": masks,
        # "sample_weights": sample_weights,
    }
    return out_batch


def _apply_mixup(batch):
    """Mixup."""
    batch = dict(**batch)
    bs = tf.shape(batch["image"])[0] // 2
    ratio = batch["mixup_ratio"][:bs, None, None, None]
    images = ratio * batch["image"][:bs] + (1.0 - ratio) * batch["image"][bs:]
    masks = ratio * batch["mask"][:bs] + (1.0 - ratio) * batch["mask"][bs:]
    # sample_weights = (
    #     ratio * batch["sample_weights"][:bs]
    #     + (1.0 - ratio) * batch["sample_weights"][bs:]
    # )

    out_batch = {
        "image": images,
        "mask": masks,
        # "sample_weights": sample_weights,
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
    # mixup_weights = batch["sample_weights"][:bs] + batch["sample_weights"][bs : 2 * bs]

    cutmix_mask = batch["cutmix_mask"][2 * bs : 3 * bs]
    cutmix_images = (
        cutmix_mask * batch["image"][2 * bs : 3 * bs]
        + (1.0 - cutmix_mask) * batch["image"][-bs:]
    )

    cutmix_masks = (
        cutmix_mask * batch["mask"][2 * bs : 3 * bs]
        + (1.0 - cutmix_mask) * batch["mask"][-bs:]
    )
    # cutmix_weights = (
    #     cutmix_mask * batch["sample_weights"][2 * bs : 3 * bs]
    #     + (1.0 - cutmix_mask) * batch["sample_weights"][-bs:]
    # )

    return {
        "image": tf.concat([mixup_images, cutmix_images], axis=0),
        "mask": tf.concat([mixup_masks, cutmix_masks], axis=0),
        # "sample_weights": tf.concat([mixup_weights, cutmix_weights], axis=0),
    }


def _add_sample_weights_to_batch(batch, bg_weight, fg_weight, use_static_weights):
    batch = dict(**batch)

    if not use_static_weights:
        bg_weight, _ = _get_dynamic_sample_weights(batch["mask"], fg_weight)

    sample_weights = tf.where(
        batch["mask"] == 0,
        bg_weight,
        fg_weight,
    )

    return {
        "image": batch["image"],
        "mask": batch["mask"],
        "sample_weights": sample_weights,
    }


@tf.function
def preprocess_ds(
    ds: tf.data.Dataset,
    augment_name: str | None,
    is_training: bool,
    batch_size: int,
    use_sample_weights: bool,
    foreground_weight: float,
    background_weight: float,
    image_shape: tuple[int, int],
    use_static_weights: bool,
    reduce_batch_by_factor: int,
    version: str,
):
    h, w = image_shape

    if augment_name is None:
        augment_name = ""

    ds = ds.repeat()

    if is_training:
        if "randcrop_" in augment_name:
            crop_size = int(augment_name.split("randcrop_")[1].split("_")[0])
            crop_fn = functools.partial(random_crop, crop_h=crop_size, crop_w=crop_size)
            ds = ds.map(crop_fn)
            h, w = crop_size, crop_size

        if "transform" in augment_name:
            transform_type = augment_name.split("transform_")[1].split("_")[0]
            aug_type = AugType.from_string(transform_type)
            aug_fn = functools.partial(augment, transform=build_aug_transform(aug_type))
            ds = ds.map(aug_fn)

        ds = ds.map(_normalize)

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
        ds = ds.map(_normalize).batch(batch_size, drop_remainder=True)

    if "patchify_" in augment_name:
        patch_size = int(augment_name.split("patchify_")[1].split("_")[0])
        patchify_fn = functools.partial(
            patchify,
            patch_size=patch_size,
            reduce_batch_by_factor=reduce_batch_by_factor,
        )
        ds = ds.map(patchify_fn)

    elif "remove_easy" in augment_name:
        
        patch_size = get_patch_size(version)
        prepatchify_im_size = get_prepatchify_resolution(version)
        num_patches_per_sample = (
            (prepatchify_im_size // patch_size) ** 2 if patch_size else 1
        )
        prepatchify_batch_size = batch_size // num_patches_per_sample

        remove_easy_fn = functools.partial(
            reduce_batch_by_remove_samples_with_less_downy,
            prepatchify_batch_size=prepatchify_batch_size,
            prepatchify_im_size=prepatchify_im_size,
            patch_size=patch_size,
            reduce_batch_by_factor=reduce_batch_by_factor,
        )
        ds = ds.map(remove_easy_fn)

    if use_sample_weights:
        add_sample_weights_to_batch = functools.partial(
            _add_sample_weights_to_batch,
            fg_weight=foreground_weight,
            bg_weight=background_weight,
            use_static_weights=use_static_weights,
        )

        ds = ds.map(add_sample_weights_to_batch)

    return ds


def prepare_ds_train(
    version,
    data_dir,
    batch_size,
    augment_name,
    foreground_weight,
    background_weight,
    use_sample_weights,
    use_static_weights,
    reduce_batch_by_factor,
) -> tuple[tf.data.Dataset, int]:
    h = w = get_resolution(version)

    ds_train, ds_info = tfds.load(
        f"downy_segmentation:{version}",
        split="train",
        data_dir=data_dir,
        download=False,
        with_info=True,
    )

    train_examples = ds_info.splits["train"].num_examples

    ds_train = ds_train.shuffle(batch_size * 10, reshuffle_each_iteration=True)  # type: ignore

    ds_train = preprocess_ds(
        ds_train,
        augment_name=augment_name,
        is_training=True,
        batch_size=batch_size,
        use_sample_weights=use_sample_weights,
        foreground_weight=foreground_weight,
        background_weight=background_weight,
        image_shape=(h, w),
        use_static_weights=use_static_weights,
        reduce_batch_by_factor=reduce_batch_by_factor,
        version=version,
    )  # type: ignore

    return (ds_train, train_examples)  # type: ignore


def prepare_ds_test(
    version,
    data_dir,
    batch_size,
    augment_name,
    foreground_weight,
    background_weight,
    use_sample_weights,
    use_static_weights,
    rebatch_ds_test_after_patchify,
) -> tuple[tf.data.Dataset, int]:
    h = w = get_resolution(version)

    ds_test, ds_info = tfds.load(
        f"downy_segmentation:{version}",
        split="test",
        data_dir=data_dir,
        download=False,
        with_info=True,
    )

    test_examples = ds_info.splits["test"].num_examples

    ds_test = ds_test.shuffle(batch_size * 10, reshuffle_each_iteration=True)  # type: ignore

    ds_test = preprocess_ds(
        ds_test,
        augment_name=augment_name,
        is_training=False,
        batch_size=batch_size,
        use_sample_weights=use_sample_weights,
        foreground_weight=foreground_weight,
        background_weight=background_weight,
        image_shape=(h, w),
        use_static_weights=use_static_weights,
        reduce_batch_by_factor=1,
        version=version,
    )  # type: ignore

    if "patchify_" in augment_name and rebatch_ds_test_after_patchify:
        ds_test = ds_test.unbatch().batch(batch_size, drop_remainder=True)  # type: ignore

    return (ds_test, test_examples)  # type: ignore


def prepare_ds(
    train_version,
    test_version,
    batch_size,
    train_augment_name: str = "",
    test_augment_name: str = "",
    foreground_weight: float = 1e3,
    background_weight: float = 0.0,
    use_sample_weights: bool = True,
    data_dir: str = "./",
    use_static_weights: bool = True,
    reduce_batch_by_factor: int = 1,
    rebatch_ds_test_after_patchify: bool = True,
):
    ds_train, train_examples = prepare_ds_train(
        version=train_version,
        data_dir=data_dir,
        batch_size=batch_size,
        augment_name=train_augment_name,
        foreground_weight=foreground_weight,
        background_weight=background_weight,
        use_sample_weights=use_sample_weights,
        use_static_weights=use_static_weights,
        reduce_batch_by_factor=reduce_batch_by_factor,
    )

    ds_test, test_examples = prepare_ds_test(
        version=test_version,
        data_dir=data_dir,
        batch_size=batch_size,
        augment_name=test_augment_name,
        foreground_weight=foreground_weight,
        background_weight=background_weight,
        use_sample_weights=use_sample_weights,
        use_static_weights=use_static_weights,
        rebatch_ds_test_after_patchify=rebatch_ds_test_after_patchify,
    )

    def unpack(batch):
        if use_sample_weights:
            return batch["image"], batch["mask"], batch["sample_weights"]
        else:
            return batch["image"], batch["mask"]

    ds_train = ds_train.map(unpack).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.map(unpack).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test, (train_examples, test_examples)


def patch(
    image,
    mask,
    # sample_weights,
    patch_size,
):
    """To be run after batching"""

    all: Any = tf.concat(
        [
            image,
            mask,
            # sample_weights
        ],
        axis=-1,
    )

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
    # sw_patches = patches[..., 4:]

    return (
        im_patches,
        m_patches,
        # sw_patches
    )


def join_patches(x: Any, batch_size: int, im_size: int) -> np.ndarray:
    b, patch_size, _, c = x.shape
    x = tf.reshape(x, (batch_size, -1, *x.shape[-3:]))
    grid_shape = int(x.shape[1] ** 0.5)
    x = tf.reshape(x, (batch_size, grid_shape, grid_shape, *x.shape[-3:])).numpy()
    placeholder = np.zeros(shape=(batch_size, im_size, im_size, c), dtype=x.dtype)

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
