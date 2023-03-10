import glob
import logging
import os
import time
from pathlib import Path

import dask.array as da
import numpy as np
import tifffile as tff
from fibsem.segmentation import model as sm

import salami.config as cfg


def run_segmentation(path: Path, checkpoint: Path = None) -> None:
    # load model
    model: sm.SegmentationModel = sm.load_model(
        encoder="resnet101", checkpoint=checkpoint, nc=7
    )

    # Note: working with single images atm
    # loop through images to imitate sequential processing
    filenames = sorted(glob.glob(os.path.join(path, "*.tif*")))

    start = time.time()
    for i, fname in enumerate(filenames):
        run_segmentation_on_image(fname, path, model)
    end = time.time()
    total = end - start
    print(f"TOTAL TIME: {total:.3f}s, {total/len(filenames):.3f}s per image")
    return


def run_segmentation_on_image(
    fname: Path, path: Path, model: sm.SegmentationModel
) -> None:
    basename = os.path.basename(fname).split(".")[0]

    t0 = time.time()

    # load image
    image = tff.imread(fname)
    t1 = time.time()

    # run inference
    mask = model.inference(image, rgb=False)

    t2 = time.time()

    # save results
    save_path = os.path.join(path, cfg.SEG_DIR, f"{basename}.tif")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tff.imsave(save_path, mask)

    t3 = time.time()

    print(
        f"MODEL_INFERENCE {basename} | LOAD {t1-t0:.3f}s | INFERENCE {t2-t1:.3f}s | SAVE {t3-t2:.3f}s | TOTAL {t3-t0:.3f}s"
    )


def calc_seg_diagnostic(path: Path):

    # load segmentation
    mask = tff.imread(os.path.join(path, cfg.SEG_DIR, "*.tif*"))

    # count number of pixels in each class in each image
    counts = np.zeros((mask.shape[0], 7))
    for i in range(mask.shape[0]):
        for j in range(7):
            counts[i, j] = np.sum(mask[i] == j)

    # calculate the percentage of each class in each image
    percentages = counts / np.sum(counts, axis=1)[:, None]

    # calculate the mean percentage of each class across all images
    mean_percentages = np.mean(percentages, axis=0)

    # calculate the standard deviation of the percentage of each class across all images
    std_percentages = np.std(percentages, axis=0)

    # calculate the mean percentage of each class across all images
    mean_counts = np.mean(counts, axis=0)

    # calculate the standard deviation of the percentage of each class across all images
    std_counts = np.std(counts, axis=0)

    print(f"Mean Counts: {mean_counts}")
    print(f"Std Counts: {std_counts}")
    print(f"Mean Percentages: {mean_percentages}")
    print(f"Std Percentages: {std_percentages}")

    import matplotlib.pyplot as plt

    # plot the counts and statistics
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].bar(np.arange(7), mean_counts, yerr=std_counts)
    ax[0].set_title("Mean Counts")
    ax[0].set_xlabel("Class")
    ax[0].set_ylabel("Count")

    ax[1].bar(np.arange(7), mean_percentages, yerr=std_percentages)
    ax[1].set_title("Mean Percentages")
    ax[1].set_xlabel("Class")
    ax[1].set_ylabel("Percentage")

    ax[2].plot(counts)
    ax[2].set_title("Counts")
    ax[2].set_xlabel("Image")
    ax[2].set_ylabel("Count")
    # add legend
    ax[2].legend(
        [
            "Background",
            "Nucleus",
            "Nucleolus",
            "Cytoplasm",
            "Mitochondria",
            "ER",
            "Golgi",
        ],
        loc="upper left",
    )

    plt.show()
