from __future__ import annotations

from typing import Any

from matplotlib import pyplot as plt
import numpy as np

# import cv2
import tensorflow as tf
import wandb

from ICAR_leaf_classification.Segmentation import load_dataset, train

# def overlay_mask(img, mask, alpha=0.3):
#     """
#     img and mask should be of the same dimensions
#     alpha should be a float between 0 and 1
#     """
#     return cv2.addWeighted(img, 0.7, mask, alpha, 0)


def display(
    display_list,
    label: str | None = None,
    log_wandb: bool = False,
    figsize: tuple = (15, 15),
    title: list[str] = ["Input Image", "True Mask", "Predicted Mask"],
):
    plt.figure(figsize=figsize)

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    if log_wandb:
        wandb.log({label: plt})
    else:
        plt.show()

def get_ds(args: train.Args):
    return load_dataset.prepare_ds(
            train_version=args.version,
            test_version=args.test_version,
            batch_size=args.batch_size,
            train_augment_name=args.train_augment_name,
            test_augment_name=args.test_augment_name,
            foreground_weight=args.fg_weight,
            background_weight=args.bg_weight,
            use_sample_weights=args.use_sample_weights,
            data_dir=args.data_dir,
            use_static_weights=args.use_static_weights,
            reduce_batch_by_factor=args.reduce_batch_by_factor,
        )

def display_dataset(
    dataset,
    num=3,
    figsize: tuple = (5, 5),
    title: list[str] = ["Image", "Mask"],
    show_sample_weights: bool = False,
):
    for images, masks, sws in dataset.unbatch().batch(num).take(1):
        for image, mask, sws in zip(images, masks, sws):
            display_list = [image, mask] + ([sws] if show_sample_weights else [])
            title = title + (["Sample Weights"] if show_sample_weights else [])
            display(display_list=display_list, figsize=figsize, title=title)


def create_mask(pred_mask: Any):
    pred_mask = tf.math.round(pred_mask)
    return pred_mask[0]


def show_predictions(model, dataset, num=1, label="pred", log_wandb: bool = False):
    for image, mask, *_ in dataset.take(num):
        pred_mask = model.predict(image)
        display([image[0], mask[0], create_mask(pred_mask)], label, log_wandb)


def show_predictions_patchify(
    model,
    patchify_ds,
    batch_size: int,
    version: str,
    num=1,
    label="pred",
    log_wandb: bool = False,
):

    for batch in patchify_ds.take(num):
        image, mask = batch["image"], batch["mask"]
        B, H, W, C = tf.shape(image)
        batch_images = tf.reshape(image, [B // batch_size, batch_size, H, W, C])
        batch_pred_masks = tf.convert_to_tensor(
            [model.predict(image) for image in batch_images]
        )
        batch_pred_masks = tf.reshape(
            batch_pred_masks, (-1, *tf.shape(batch_pred_masks)[2:])
        )

        im_size = load_dataset.get_prepatchify_resolution(version)
        image = load_dataset.join_patches(image, batch_size=1, im_size=im_size)
        mask = load_dataset.join_patches(mask, batch_size=1, im_size=im_size)
        pred_mask = load_dataset.join_patches(
            batch_pred_masks, batch_size=1, im_size=im_size
        )

        display([image[0], mask[0], create_mask(pred_mask)], label, log_wandb)


class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        dataset: tf.data.Dataset,
        args: train.Args,
        display_every: int = 1,
        log_wandb: bool = False,
    ):
        super().__init__()
        self.display_every = display_every
        self.dataset = dataset
        self.log_wandb = log_wandb
        self.args = args

        if "patchify_" in self.args.train_augment_name:

            self.patchify_ds, _ = load_dataset.prepare_ds_test(
                version=args.test_version,
                data_dir=args.data_dir,
                batch_size=1,
                augment_name=args.test_augment_name,
                foreground_weight=args.fg_weight,
                background_weight=args.bg_weight,
                use_sample_weights=args.use_sample_weights,
                use_static_weights=args.use_static_weights,
                rebatch_ds_test_after_patchify=False,
            )

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.display_every == 0:
            self.show_n_pred(f"ep: {epoch+1}")

    def show_n_pred(self, label, num=1, log_wandb: bool | None = None):

        if log_wandb is None:
            log_wandb = self.log_wandb

        if "patchify_" in self.args.train_augment_name:
            show_predictions_patchify(
                self.model,
                patchify_ds=self.patchify_ds,
                batch_size=1,
                version=self.args.version,
                label=label,
                log_wandb=log_wandb,
                num=num,
            )
        else:
            show_predictions(
                self.model,
                self.dataset,
                label=label,
                log_wandb=log_wandb,
                num=num,
            )


def show_schedule(lr_schedule, total_steps):
    points = []
    for step in range(total_steps):
        points.append([step, lr_schedule(step)])
    points = np.array(points)
    plt.plot(points[:, 0], points[:, 1])


def plot_model_history(model_history: Any, log_wandb: bool = False) -> None:
    loss = model_history.history["loss"]
    val_loss = model_history.history["val_loss"]

    plt.figure()
    plt.plot(model_history.epoch, loss, "r", label="Training loss")
    plt.plot(model_history.epoch, val_loss, "bo", label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.ylim([0, 10])
    plt.legend()
    if log_wandb:
        wandb.log({"model_history": plt})
    else:
        plt.show()
