import glob
import logging
import os
import time
from pathlib import Path

import dask.array as da
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
        tff.imsave(save_path, mask)

        t3 = time.time()

        print(
            f"MODEL_INFERENCE | LOAD {t1-t0:.3f}s | INFERENCE {t2-t1:.3f}s | SAVE {t3-t2:.3f}s | TOTAL {t3-t0:.3f}s"
        )

    end = time.time()
    total = end - start
    print(f"TOTAL TIME: {total:.3f}s, {total/len(filenames):.3f}s per image")
    return
