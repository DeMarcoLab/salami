import os
import random
import time

from fibsem import utils

import salami
import salami.config as cfg
import salami.segmentation.segmentation as sseg
from salami import analysis as sa
from salami import core as sc
from salami.structures import SalamiSettings

def analysis_pipeline():

    sa.run_salami_analysis(break_idx=1)


def full_pipeline():

    # connect to microscope, load settings
    microscope, settings = utils.setup_session(protocol_path=cfg.PROTOCOL_PATH)
    
    # create output directory
    # path = os.path.join(settings.image.save_path, cfg.DATA_DIR)
    path ="/home/patrick/github/salami/salami/output"

    RAW_PATH = os.path.join(path, cfg.RAW_DIR)
    DENOISE_PATH = os.path.join(path, cfg.DENOISE_DIR)
    SEG_PATH = os.path.join(path, cfg.SEG_DIR)

    os.makedirs(RAW_PATH, exist_ok=True)
    os.makedirs(DENOISE_PATH, exist_ok=True)
    os.makedirs(SEG_PATH, exist_ok=True)
    settings.image.save_path = RAW_PATH

    # load salami protocol
    ss = sc.load_protocol(settings.protocol)

    # run salami
    # sc.run_salami(microscope, settings, ss)

    # run denoising
    print("Running denoising model...")
    from salami.denoise import inference
    inference.run_denoise(RAW_PATH, DENOISE_PATH, settings.protocol["denoise"])
    
    # run segmentation
    print("Running segmentation model...")
    sseg.run_segmentation(DENOISE_PATH, checkpoint=None, output_path=SEG_PATH)

    # diagnostics
    sseg.calc_seg_diagnostic(SEG_PATH, labels=cfg.LABELS)

    print("Done!")


def main():

    full_pipeline()

    # analysis_pipeline()


if __name__ == "__main__":
    main()
