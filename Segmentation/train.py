from __future__ import annotations


from dataclasses import dataclass
import os
import sys
from matplotlib import pyplot as plt
import numpy as np

os.environ["SM_FRAMEWORK"] = "tf.keras"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import segmentation_models as sm
from Segmentation import load_dataset, utils, train

import rich
import wandb
from absl import app
from etils import eapp
from etils.etree import tree as etree


@dataclass
class Args:
    run_name: str = "test"
    backbone: str = "efficientnetb0"
    version: str = "7.2048.256"
    test_version: str | None = None
    train_augment_name: str = "cutmix_mixup_0.4"
    test_augment_name: str | None = None
    fg_weight: float = 1.0  # Needs to be set for both dynamic and static weights
    bg_weight: float = 1e-8
    batch_size: int = 8
    epochs: int = 30
    warmup_epochs: int = 5
    lr: float = 1e-3
    use_lr_schedule: bool = True
    run_eagerly: bool = False
    use_wandb: bool = True
    use_sample_weights: bool = True
    data_dir: str = "/content"
    use_static_weights: bool = False
    check_if_exp_already_ran: bool = True
    early_stopping_patience: int = 30
    reduce_batch_by_factor: int = 1

    def __post_init__(self):
        self.actual_batch_size = self.batch_size
        self.step_per_batch = self.batch_size

        if self.test_version is None:
            self.test_version = self.version

        if self.test_augment_name is None:
            self.test_augment_name = ""
            if "patchify_" in self.train_augment_name:
                patch_size = int(
                    self.train_augment_name.split("patchify_")[1].split("_")[0]
                )
                self.test_augment_name = f"patchify_{patch_size}"
            elif "randcrop_" in self.train_augment_name:
                crop_size = int(
                    self.train_augment_name.split("randcrop_")[1].split("_")[0]
                )
                self.test_augment_name = f"patchify_{crop_size}"

        if "patchify_" in self.train_augment_name:
            patch_size = int(
                self.train_augment_name.split("patchify_")[1].split("_")[0]
            )
            prepatchify_im_size = load_dataset.get_resolution(self.version)
            num_patches_per_sample = (
                (prepatchify_im_size // patch_size) ** 2 if patch_size else 1
            )
            self.actual_batch_size = self.batch_size * num_patches_per_sample

        if "remove_easy" in self.train_augment_name:
            self.actual_batch_size = (
                self.actual_batch_size // self.reduce_batch_by_factor
            )
            self.step_per_batch = self.batch_size

        if self.reduce_batch_by_factor > 1:
            # Check if either patchify or randcrop or remove_easy are in the augment name
            if not any(
                [
                    "patchify_" in self.train_augment_name,
                    "randcrop_" in self.train_augment_name,
                    "remove_easy" in self.train_augment_name,
                ]
            ):
                self.reduce_batch_by_factor = 1
                print(
                    "Setting reduce_batch_by_factor to 1 as"
                    " no patchify or randcrop or remove_easy in augment name"
                )
        if "transform" in self.train_augment_name:
            transform_type = self.train_augment_name.split("transform_")[1].split("_")[
                0
            ]
            print(f"{transform_type=}")
            if transform_type.upper() not in load_dataset.AugType.__members__:
                print(f"Invalid transform type: {transform_type} in train augment name")
                self.train_augment_name = self.train_augment_name.replace(
                    "transform", "transform_max"
                )
                print(f"Setting train_augment_name to: {self.train_augment_name}")

        print(f"{self.actual_batch_size=}")


def build_model(args: Args):
    if args.use_lr_schedule:
        total_train_images = load_dataset.get_total_train_images(args.version)
        STEPS_PER_EPOCH = total_train_images // args.step_per_batch
        lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.0,
            decay_steps=int(args.epochs * STEPS_PER_EPOCH),
            warmup_target=args.lr,
            warmup_steps=int(args.warmup_epochs * STEPS_PER_EPOCH),
        )
    else:
        lr = args.lr

    model = sm.Unet(args.backbone, encoder_weights="imagenet")
    model.compile(
        tf.keras.optimizers.Adam(learning_rate=lr),
        loss=sm.losses.bce_jaccard_loss,
        run_eagerly=args.run_eagerly,
        weighted_metrics=[
            sm.metrics.iou_score,
            sm.metrics.IOUScore(threshold=0.5, name="bin_iou"),
        ]
        if args.use_sample_weights
        else [],
        metrics=[
            sm.metrics.iou_score,
            sm.metrics.IOUScore(threshold=0.5, name="bin_iou"),
        ]
        if not args.use_sample_weights
        else [],
    )
    return model


def build_callbacks(args: Args):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.early_stopping_patience,
        restore_best_weights=True,
    )

    log_wandb = wandb.keras.WandbCallback(
        save_graph=(False),
        save_model=(False),
        monitor="val_bin_iou",
        mode="max",
    )

    res = load_dataset.get_resolution(args.version)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"./models/unet/{args.backbone}_{res}_{args.train_augment_name}_{args.run_name}",
        monitor="val_loss",
        verbose=0,
        save_weights_only=True,
        save_best_only=True,
    )

    callbacks = [early_stopping, log_wandb, model_checkpoint]

    if not args.use_lr_schedule:
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3
        )
        callbacks.append(reduce_lr)

    return callbacks


def sanity_check_dataset(args: Args, num_display: int = 5):
    train_ds, test_ds = load_dataset.prepare_ds(  # type: ignore
        train_version=args.version,
        test_version=args.test_version,
        batch_size=args.batch_size,
        train_augment_name=args.train_augment_name,
        test_augment_name=args.test_augment_name,  # type: ignore
        foreground_weight=args.fg_weight,
        background_weight=args.bg_weight,
        use_sample_weights=args.use_sample_weights,
        data_dir=args.data_dir,
        use_static_weights=args.use_static_weights,
        reduce_batch_by_factor=args.reduce_batch_by_factor,
        rebatch_ds_test_after_patchify=False,
    )

    for data in train_ds.take(num_display):
        etree.spec_like(data)  # type: ignore

    for data in test_ds.take(num_display):
        etree.spec_like(data)  # type: ignore


def train_exp(args: Args):
    rich.print(args)
    res = load_dataset.get_prepatchify_resolution(args.version)

    wandb.login(key="a970489934de52f5c9883188b98f55afd4e56ed7")

    run_name = f"{args.run_name}_{args.backbone}_{res}"

    if args.train_augment_name:
        run_name += f"_{args.train_augment_name}"

    if args.use_sample_weights:
        if args.use_static_weights:
            run_name += f"_fg={args.fg_weight}_bg={args.bg_weight}"
        else:
            run_name += f"_fg={args.fg_weight}_bg=dynamic"

    run_name += (
        f"_lr={args.lr}_bs={args.batch_size}_epochs={args.epochs}_v={args.version}"
    )

    if args.reduce_batch_by_factor > 1:
        run_name += f"_rf={args.reduce_batch_by_factor}"

    entity = "icar"
    project = "downy_segmentation"

    if args.check_if_exp_already_ran:
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}")
        if any(run_name == run.name for run in runs):
            print(f"Run: {run_name} already ran. Skipping...")
            return

    wandb.init(
        project=project,
        entity=entity,
        reinit=True,
        name=run_name,
    )

    (
        train_ds,
        test_ds,
        (total_train_images, total_test_images),
    ) = load_dataset.prepare_ds(
        train_version=args.version,
        test_version=args.test_version,
        batch_size=args.batch_size,
        train_augment_name=args.train_augment_name,
        test_augment_name=args.test_augment_name,  # type: ignore
        foreground_weight=args.fg_weight,
        background_weight=args.bg_weight,
        use_sample_weights=args.use_sample_weights,
        data_dir=args.data_dir,
        use_static_weights=args.use_static_weights,
        reduce_batch_by_factor=args.reduce_batch_by_factor,
    )

    # total_train_images = load_dataset.get_total_train_images(args.version)
    # total_test_images = load_dataset.get_total_test_images(args.version)

    STEPS_PER_EPOCH = total_train_images // args.step_per_batch
    VAL_STEPS_PER_EPOCH = total_test_images // args.step_per_batch

    print("Building model")
    model = train.build_model(args)

    display_callback = utils.DisplayCallback(test_ds, args, display_every=1)

    print("Building callbacks")
    callbacks = [
        *train.build_callbacks(args),
        display_callback,
    ]

    model_history = model.fit(
        train_ds,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=test_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VAL_STEPS_PER_EPOCH,
        callbacks=callbacks,
    )

    utils.plot_model_history(model_history, log_wandb=True)

    print("Visualizing results")
    utils.show_predictions(model, train_ds, 5, log_wandb=True, label=f"train")
    display_callback.show_n_pred(label=f"test", num=5, log_wandb=True)




def fix_sample_weight():
    """RESULTS -> ...."""

    for use_sample_weight in [False, True]:
        train_exp(
            Args(
                run_name=f"FixSampleWeight",
                backbone="efficientnetb0",
                version="7.512.0",
                train_augment_name="",
                batch_size=8,
                epochs=6,
                early_stopping_patience=6,
                warmup_epochs=1,
                use_sample_weights=use_sample_weight,
                use_static_weights=False,
                fg_weight=1.0,
            )
        )


def RezImproves():
    """
    RESULTS ->
        Clear signs of improving with increasing resolution.
        Overfitting also rampant.
    """

    TRAIN_VERSIONS = {
        "7.256.0": 8,
        "7.512.0": 4,
        "7.1024.0": 2,
    }

    for train_version, batch_size in TRAIN_VERSIONS.items():
        train_exp(
            Args(
                run_name=f"RezImproves",
                backbone="efficientnetb2",
                version=train_version,
                train_augment_name="",
                batch_size=batch_size,
                epochs=6,
                early_stopping_patience=6,
                warmup_epochs=1,
                use_sample_weights=True,
                use_static_weights=False,
            )
        )


def find_best_transform():
    
    AUGMENTATIONS = {
        "",
        "transform_light",
        "transform_med",
        "transform_heavy",
        "transform_max",
    }

    for augment_name in AUGMENTATIONS:
        train_exp(
            Args(
                run_name=f"BestTransform2",
                backbone="efficientnetb2",
                version="7.1024.0",
                train_augment_name=augment_name + "_patchify_256",
                test_version="7.1024.0",
                test_augment_name=f"patchify_256",
                batch_size=2,
                epochs=6,
                warmup_epochs=1,
                lr=1e-3,
                use_sample_weights=True,
                fg_weight=1.0,
                use_static_weights=False,
                early_stopping_patience=6,
                reduce_batch_by_factor=1,
            )
        )

def final_experiments():
    
    train_exp(
        Args(
            run_name=f"Best1",
            backbone="efficientnetb2",
            version="7.2048.0",
            train_augment_name="transform_max_cutmix_mixup_0.4" + "_patchify_512" + "_remove_easy",
            test_version="7.2048.0",
            test_augment_name=f"patchify_512",
            batch_size=1,
            epochs=6,
            warmup_epochs=1,
            lr=5e-4,
            use_sample_weights=True,
            fg_weight=1.0,
            use_static_weights=False,
            early_stopping_patience=6,
            reduce_batch_by_factor=1,
        )
    )


def show_useful_augs():
    
    AUGMENTATIONS = {
        "cutmix",
        "mixup_0.4",
    }
    
    for augment_name in AUGMENTATIONS:
        train_exp(
            Args(
                run_name=f"ShowUsefulAugs",
                backbone="efficientnetb2",
                version="7.1024.0",
                train_augment_name=augment_name + "_patchify_256",
                test_version="7.1024.0",
                test_augment_name=f"patchify_256",
                batch_size=2,
                epochs=6,
                warmup_epochs=1,
                lr=1e-3,
                use_sample_weights=True,
                fg_weight=1.0,
                use_static_weights=False,
                early_stopping_patience=6,
                reduce_batch_by_factor=1,
            )
        )


def PatchifyEquivalence():
    """
    RESULTS ->
        7.512.256 and 8.512.256 are not equivalent.
            (Makes sense, I am removing ton of information from the dataset)
        9.512.256 is equivalent to 7.512.0.
        TODO: Find out whether 9.512.256 with remove_easy is equivalent to 7.512.0
        9.1024.256 is not equivalent to 7.1024.0, dropping this idea, in
            favor of 7.1024.0 with patchify_256.
        Prove that 7.1024.0 with patchify_256, remove_easy_2 and 7.1024.0 with
            patchfiy_512, remove_easy_2 are equivalent to 7.1024.0.

    """
    # TRAIN_VERSIONS = {
    #     "7.512.0": [
    #         # "7.512.256",  # Not Equivalent
    #         # "8.512.256",  # Not equivalent
    #         # "9.512.256",  # Equivalent, Not equivalent with remove_easy
    #     ],
    #     "7.1024.0": [
    #         # "7.1024.256",  # Not Equivalent
    #         # "8.1024.256",  # Not equivalent
    #         # "9.1024.256",  # Not equivalent
    #     ],
    # }

    REDUCE_FACTORS = [1, 2]
    PATCHIFY_RES = [512]
    for patch_size in PATCHIFY_RES:
        for rf in REDUCE_FACTORS:
            train_exp(
                Args(
                    run_name=f"ShowPatchifyEq",
                    backbone="efficientnetb2",
                    version="7.1024.0",
                    train_augment_name=f"patchify_{patch_size}"
                    + ("_remove_easy" if rf > 1 else ""),
                    test_version="7.1024.0",
                    test_augment_name=f"patchify_{patch_size}",
                    batch_size=2 * rf,
                    epochs=6 * rf,
                    early_stopping_patience=6 * rf,
                    warmup_epochs=1 * rf,
                    use_sample_weights=True,
                    use_static_weights=False,
                    reduce_batch_by_factor=rf,
                    lr=0.001,
                )
            )

    # train_exp(
    #         Args(
    #             run_name=f"SimplePatchify",
    #             backbone="efficientnetb2",
    #             version="7.1024.0",
    #             train_augment_name="patchify_256",
    #             test_version="7.1024.0",
    #             test_augment_name=f"patchify_256",
    #             batch_size=2,
    #             epochs=6,
    #             early_stopping_patience=6,
    #             warmup_epochs=1,
    #             use_sample_weights=True,
    #             use_static_weights=False,
    #         )
    #     )
    

def show_need_sample_weights():
    for use_sample_weight in [False, True]:
        if use_sample_weight:
            use_static_weights = [False, True]
            sw_pair_list = [
                [(1, 1), (10, 1), (1000, 1)],
                [(1, 0), (10, 0), (10, 1), (1000, 1)],
            ]
        else:
            sw_pair_list = [[(1, 1)]]
            use_static_weights = [False]

        for use_sw in use_static_weights:
            sw_pairs = sw_pair_list[int(use_sw)]

            for fg_w, bg_w in sw_pairs:
                train_exp(
                    Args(
                        run_name=f"ShowNeedSW",
                        backbone="efficientnetb0",
                        version="7.512.0",
                        train_augment_name="",
                        batch_size=8,
                        epochs=30,
                        lr=1e-3,
                        use_sample_weights=use_sample_weight,
                        fg_weight=fg_w,
                        bg_weight=bg_w,
                        use_static_weights=use_sw,
                    )
                )


def show_need_big_models():
    BACKBONES = [
        "efficientnetb7",
        "efficientnetb4",
        "efficientnetb2",
        "efficientnetb0",
    ]

    for BACKBONE in BACKBONES:
        train_exp(
            Args(
                run_name=f"ShowNeedBigModels",
                backbone=BACKBONE,
                version="8.2048.256",
                train_augment_name="mixup_0.4",
                test_version="7.2048.0",
                test_augment_name="patchify_256",
                batch_size=8,
                epochs=5,
                warmup_epochs=1,
                lr=1e-3,
                use_sample_weights=True,
                fg_weight=1.0,
                use_static_weights=False,
                early_stopping_patience=10,
            )
        )


def show_ds_equivalence():
    TRAIN_VERSIONS = [
        "7.512.0",
        "7.512.256",
        "8.512.256",
    ]

    for _version in TRAIN_VERSIONS:
        train_aug = "patchify_256" if "256" not in _version else ""
        reduce_batch_by_factor = 2 if "256" not in _version else 1
        train_exp(
            Args(
                run_name=f"ShowDSEquivalence2",
                backbone="efficientnetb0",
                version=_version,
                train_augment_name=train_aug,
                test_version="7.512.0",
                test_augment_name="patchify_256",
                batch_size=4,
                epochs=30,
                warmup_epochs=3,
                early_stopping_patience=30,
                reduce_batch_by_factor=reduce_batch_by_factor,
            )
        )


def show_need_aug():
    AUGMENTATIONS = {
        "",
        "transform",
        "cutmix",
        "mixup_0.4",
        "cutmix_mixup_0.4",
        "transform_cutmix_mixup_0.4",
    }

    for augment_name in AUGMENTATIONS:
        train_exp(
            Args(
                run_name=f"ShowNeedAug",
                backbone="efficientnetb2",
                version="7.1024.0",
                train_augment_name=augment_name + "_patchify_256",
                batch_size=1,
                epochs=50,
                lr=1e-3,
                use_sample_weights=True,
                fg_weight=1.0,
                use_static_weights=False,
                early_stopping_patience=50,
                reduce_batch_by_factor=2,
            )
        )


def light_sweep_t4():
    BACKBONES = [
        "efficientnetb0",
        "efficientnetb4",
    ]

    VERSIONS = ["7.256.0", "7.512.0", "7.1024.0", "7.2048.0"]
    AUGMENTATIONS = {
        "",
        "transform",
        "cutmix",
        "mixup_0.4",
        "cutmix_mixup_0.4",
        "transform_cutmix_mixup_0.4",
    }

    PATCHIFY_RES = [256, 512]

    BATCH_SIZE = 2

    for BACKBONE in BACKBONES:
        for version in VERSIONS:
            for augment_name in AUGMENTATIONS:
                run_name = f"LightSweepT4"

                res = load_dataset.get_resolution(version)
                PATCHIFY_BREAK_FLAG = False
                for patch_size in PATCHIFY_RES:
                    if res > patch_size and res > PATCHIFY_RES[-1]:
                        augment_name += f"_patchify_{patch_size}"
                    else:
                        PATCHIFY_BREAK_FLAG = True

                    train_exp(
                        Args(
                            run_name=run_name,
                            backbone=BACKBONE,
                            version=version,
                            train_augment_name=augment_name,
                            batch_size=BATCH_SIZE,
                            epochs=100,
                            lr=1e-2,
                            use_sample_weights=True,
                            use_static_weights=False,
                            early_stopping_patience=10,
                        )
                    )

                    if PATCHIFY_BREAK_FLAG:
                        break


def light_sweep_for_aug_t4():
    BACKBONES = [
        "efficientnetb0",
    ]

    RESOLUTIONS = {"1024": "4.0.0"}
    AUGMENTATIONS = {
        "",
        "transform",
        "cutmix",
        "mixup_0.4",
        "cutmix_mixup_0.4",
        "transform_cutmix_mixup_0.4",
    }

    BATCH_SIZE = 2

    for BACKBONE in BACKBONES:
        for resolution, version in RESOLUTIONS.items():
            for augment_name in AUGMENTATIONS:
                run_name = f"AugSweep"
                rich.print(run_name)

                train_exp(
                    Args(
                        run_name=run_name,
                        backbone=BACKBONE,
                        version=version,
                        train_augment_name=augment_name,
                        batch_size=BATCH_SIZE,
                        epochs=50,
                        lr=1e-2,
                        use_sample_weights=True,
                        use_static_weights=False,
                    )
                )


def light_sweep_for_models_t4():
    BACKBONES = [
        "efficientnetb0",
        "resnet34",
        "mobilenetv2",
    ]

    RESOLUTIONS = {"1024": "4.0.0"}
    AUGMENTATIONS = {
        "transform_cutmix_mixup_0.4",
    }

    BATCH_SIZE = 2

    for BACKBONE in BACKBONES:
        for resolution, version in RESOLUTIONS.items():
            for augment_name in AUGMENTATIONS:
                run_name = f"ModelSweep"
                rich.print(run_name)

                train_exp(
                    Args(
                        run_name=run_name,
                        backbone=BACKBONE,
                        version=version,
                        train_augment_name=augment_name,
                        batch_size=BATCH_SIZE,
                        epochs=50,
                        lr=1e-3,
                        use_sample_weights=True,
                        use_static_weights=False,
                    )
                )


def light_sweep_for_res_t4():
    BACKBONES = [
        "efficientnetb0",
    ]

    RESOLUTIONS = {
        # "256": "5.0.0",  # Done
        "512": "3.0.0",
        "1024": "4.0.0",
    }
    AUGMENTATIONS = {
        "transform_cutmix_mixup_0.4",
    }

    BATCH_SIZE = 2

    for BACKBONE in BACKBONES:
        for resolution, version in RESOLUTIONS.items():
            for augment_name in AUGMENTATIONS:
                run_name = f"ResSweep"
                rich.print(run_name)

                train_exp(
                    Args(
                        run_name=run_name,
                        backbone=BACKBONE,
                        version=version,
                        train_augment_name=augment_name,
                        batch_size=BATCH_SIZE,
                        epochs=50,
                        lr=1e-3,
                        use_sample_weights=True,
                        use_static_weights=False,
                    )
                )


def med_sweep_t4(args: Args):
    BACKBONES = [
        "efficientnetb4",
        "resnet50",
    ]

    RESOLUTIONS = {"256": "5.0.0", "512": "3.0.0"}
    AUGMENTATIONS = {
        "transform",
        "cutmix",
        "mixup_0.4",
        "cutmix_mixup_0.4",
        "transform_cutmix_mixup_0.4",
    }

    BATCH_SIZE = 2

    for BACKBONE in BACKBONES:
        for resolution, version in RESOLUTIONS.items():
            for augment_name in AUGMENTATIONS:
                run_name = f"{args.run_name}_{BACKBONE}_{resolution}_{augment_name}"
                rich.print(run_name)

                train_exp(
                    Args(
                        run_name=run_name,
                        backbone=BACKBONE,
                        version=version,
                        train_augment_name=augment_name,
                        batch_size=BATCH_SIZE,
                        epochs=5,
                        lr=1e-3,
                        fg_weight=10,
                        bg_weight=1e-8,
                        use_sample_weights=True,
                    )
                )


def light_sweep_for_sample_weights_t4(
    data_dir: str | None = None,
    version: str = "7.256.0",
):
    BACKBONES = [
        "efficientnetb0",
    ]

    VERSIONS = [
        version,
    ]

    AUGMENTATIONS = [
        "",  # No augmentations
    ]

    foreground_weights = [1, 10, 1000]
    background_weights = [0, 1e-8, 1]

    BATCH_SIZE = 8

    LEARNING_RATES = [
        1e-2,
    ]

    data_dir = data_dir or "/content"

    for lr in LEARNING_RATES:
        for backbone in BACKBONES:
            for version in VERSIONS:
                for augment_name in AUGMENTATIONS:
                    for _fg_w in foreground_weights:
                        train_exp(
                            Args(
                                run_name=f"Sweep_sw",
                                backbone=backbone,
                                version=version,
                                train_augment_name=augment_name,
                                fg_weight=_fg_w,
                                batch_size=BATCH_SIZE,
                                epochs=50,
                                lr=lr,
                                use_sample_weights=True,
                                use_static_weights=False,
                                data_dir=data_dir,
                            )
                        )

                        for _bg_w in background_weights:
                            train_exp(
                                Args(
                                    run_name=f"Sweep_sw",
                                    backbone=backbone,
                                    version=version,
                                    train_augment_name=augment_name,
                                    fg_weight=_fg_w,
                                    bg_weight=_bg_w,
                                    batch_size=BATCH_SIZE,
                                    epochs=50,
                                    lr=lr,
                                    use_sample_weights=True,
                                    use_static_weights=True,
                                    data_dir=data_dir,
                                )
                            )


def med_sweep_V100(args: Args):
    BACKBONES = [
        "efficientnetb4",
        "resnet50",
    ]

    RESOLUTIONS = {"256": "5.0.0", "512": "3.0.0", "1024": "4.0.0", "2048": "6.0.0"}
    AUGMENTATIONS = {
        "transform",
        "cutmix",
        "mixup_0.4",
        "cutmix_mixup_0.4",
        "transform_cutmix_mixup_0.4",
    }

    BATCH_SIZE = 2

    for BACKBONE in BACKBONES:
        for resolution, version in RESOLUTIONS.items():
            for augment_name in AUGMENTATIONS:
                run_name = f"{args.run_name}_{BACKBONE}_{resolution}_{augment_name}"
                rich.print(run_name)

                train_exp(
                    Args(
                        run_name=run_name,
                        backbone=BACKBONE,
                        version=version,
                        train_augment_name=augment_name,
                        batch_size=BATCH_SIZE,
                        epochs=5,
                        lr=1e-3,
                        fg_weight=10,
                        bg_weight=1e-8,
                        use_sample_weights=True,
                    )
                )


def main_wrapper(args: Args):
    train_exp(args)


if __name__ == "__main__":
    app.run(main_wrapper, flags_parser=eapp.make_flags_parser(Args))  # type: ignore
