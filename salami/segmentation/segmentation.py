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

def load_model(checkpoint: str):
    # load model
    model: sm.SegmentationModel = sm.load_model(
        encoder="resnet101", checkpoint=checkpoint, nc=7
    )

    return model

def run_segmentation(path: Path, checkpoint: Path, output_path: Path) -> None:

    model = load_model(checkpoint)
    # Note: working with single images atm
    # loop through images to imitate sequential processing
    filenames = sorted(glob.glob(os.path.join(path, "*.tif*")))

    start = time.time()
    for i, fname in enumerate(filenames):
        run_segmentation_step(model = model, fname=fname, output_path=output_path)
    end = time.time()
    total = end - start
    print(f"TOTAL TIME: {total:.3f}s, {total/len(filenames):.3f}s per image")
    return


def run_segmentation_step(model: sm.SegmentationModel,
    fname: Path, output_path: Path
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
    save_path = os.path.join(output_path, f"{basename}.tif")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tff.imsave(save_path, mask.squeeze().astype(np.uint8))
    
    t3 = time.time()

    print(
        f"MODEL_INFERENCE {basename} | LOAD {t1-t0:.3f}s | INFERENCE {t2-t1:.3f}s | SAVE {t3-t2:.3f}s | TOTAL {t3-t0:.3f}s"
    )


def calc_seg_diagnostic(path: Path, labels: list[str] = None, plot: bool = False):

    if labels is None:
        labels = cfg.LABELS

    # load segmentation
    import dask_image.imread
    seg_path = os.path.join(path, "*.tif*")
    mask = dask_image.imread.imread(seg_path)
    filenames = sorted(glob.glob(seg_path))

    # count number of pixels in each class in each image
    counts = np.zeros((mask.shape[0], 7))
    for i in range(mask.shape[0]):
        for j in range(7):
            counts[i, j] = np.sum(mask[i] == j)

    return {"counts": counts, "filenames": filenames, "labels": labels}

    # # calculate the percentage of each class in each image
    # percentages = counts / np.sum(counts, axis=1)[:, None]

    # # calculate the mean percentage of each class across all images
    # mean_percentages = np.mean(percentages, axis=0)

    # # calculate the standard deviation of the percentage of each class across all images
    # std_percentages = np.std(percentages, axis=0)

    # # calculate the mean percentage of each class across all images
    # mean_counts = np.mean(counts, axis=0)

    # # calculate the standard deviation of the percentage of each class across all images
    # std_counts = np.std(counts, axis=0)

    # print(f"Mean Counts: {mean_counts}")
    # print(f"Std Counts: {std_counts}")
    # print(f"Mean Percentages: {mean_percentages}")
    # print(f"Std Percentages: {std_percentages}")

    # if plot:
    #     plot_seg_diagnostic(mean_counts, std_counts, mean_percentages, std_percentages, counts, labels)

    # return {"mean_counts": mean_counts, 
    #         "std_counts": std_counts, 
    #         "mean_percentages": mean_percentages, 
    #         "std_percentages": std_percentages, 
    #         "counts": counts, "labels": labels, "filenames": filenames}



# def plot_seg_diagnostic(mean_counts, std_counts, mean_percentages, std_percentages, counts, labels):
#     import matplotlib.pyplot as plt

#     # plot the counts and statistics
#     fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#     ax[0].bar(np.arange(7), mean_counts, yerr=std_counts)
#     ax[0].set_title("Mean Counts")
#     ax[0].set_xlabel("Class")
#     ax[0].set_ylabel("Count")

#     ax[1].bar(np.arange(7), mean_percentages, yerr=std_percentages)
#     ax[1].set_title("Mean Percentages")
#     ax[1].set_xlabel("Class")
#     ax[1].set_ylabel("Percentage")

#     ax[2].plot(counts)
#     ax[2].set_title("Counts")
#     ax[2].set_xlabel("Image")
#     ax[2].set_ylabel("Count")
#     # add legend
#     ax[2].legend(labels=labels, loc="upper left")

#     plt.show()


def calculate_diag_df(stats:dict):
    import pandas as pd

    df = pd.DataFrame(stats["counts"], columns=stats["labels"])

    df["fname"] = stats["filenames"]
    df["fname"] = df["fname"].apply(lambda x: os.path.basename(x))


    # calculate as a percentage
    # df.iloc[:, :-1] = df.iloc[:, :-1].div(df.iloc[:, :-1].sum(axis=1), axis=0)

    return df
