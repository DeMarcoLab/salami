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
    path = os.path.join(settings.image.save_path, cfg.DATA_DIR)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, cfg.DENOISE_DIR), exist_ok=True)
    os.makedirs(os.path.join(path, cfg.SEG_DIR), exist_ok=True)
    settings.image.save_path = path

    # load salami protocol
    ss = sc.load_protocol(settings.protocol)
    # sc.run_salami(microscope, settings, ss)


    # time.sleep(random.randint(1, 2))
    # print("Loading data...")
    # time.sleep(random.randint(1, 2))
    # print("Preprocessing data...")
    # time.sleep(random.randint(1, ))
    # print("Running denoising model...")
    # time.sleep(random.randint(1, 5))
    # print("Saving results...")
    # time.sleep(random.randint(1, 5))
    # print("Restacking and aligning arrays...")
    # time.sleep(random.randint(1, 5))
    
    print("Running segmentation model...")
    path = "/home/patrick/github/salami/salami/output/denoise"
    sseg.run_segmentation(path)
    sseg.calc_seg_diagnostic(path)

    print("Done!")


def main():

    full_pipeline()

    # analysis_pipeline()


if __name__ == "__main__":
    main()
