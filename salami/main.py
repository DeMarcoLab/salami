import os
import random
import time

from fibsem import utils
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    FibsemMillingSettings,
    FibsemPattern,
    FibsemPatternSettings,
    MicroscopeSettings,
)

import salami
import salami.config as cfg
import salami.segmentation.segmentation as ss
from salami import analysis as sa
from salami.structures import SalamiSettings


def run_collection(microscope: FibsemMicroscope, settings: MicroscopeSettings):

    # image settings
    settings.image.save = True
    settings.image.autocontrast = False
    settings.image.gamma_enabled = False

    # pattern settings
    ps = FibsemPatternSettings.__from_dict__(settings.protocol["milling"]["pattern"])

    ms = FibsemMillingSettings.__from_dict__(settings.protocol["milling"])

    # salami settings
    ss = SalamiSettings(
        n_steps=int(settings.protocol["num_steps"]),
        step_size=float(settings.protocol["step_size"]),
    )

    salami.core.run_salami(microscope, settings, ss, ps, ms)


def analysis_pipeline():

    microscope, settings = utils.setup_session(protocol_path=cfg.PROTOCOL_PATH)

    sa.run_salami_analysis(microscope, settings, "data")


def full_pipeline():

    # TODO: protocol
    microscope, settings = utils.setup_session(protocol_path=cfg.PROTOCOL_PATH)

    # sa.run_salami_analysis(microscope, settings, "data")

    path = os.path.join(settings.image.save_path, cfg.DATA_DIR)
    settings.image.save_path = path
    run_collection(microscope, settings)

    # path = "/home/patrick/github/salami/demo_2023-03-02-06-32-14PM/data"
    # create output directory
    os.makedirs(os.path.join(path, cfg.DENOISE_DIR), exist_ok=True)
    os.makedirs(os.path.join(path, cfg.SEG_DIR), exist_ok=True)

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
    # ss.run_segmentation(path)
    # ss.calc_seg_diagnostic(path)

    print("Done!")


def main():

    # full_pipeline()

    analysis_pipeline()


if __name__ == "__main__":
    main()
