
import os
from fibsem import utils
import random
import time
import salami
from salami.segmentation.segmentation import run_segmentation
from salami import analysis as sa
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import MicroscopeSettings, FibsemPatternSettings, FibsemMillingSettings, FibsemPattern
from salami.structures import SalamiSettings
import salami.config as cfg

def run_collection(microscope: FibsemMicroscope, settings: MicroscopeSettings):

    # image settings
    settings.image.save = True
    settings.image.autocontrast = False
    settings.image.gamma_enabled = False

    # milling settings
    start_x, end_x = -8e-6, 33e-6
    start_y, end_y = -4e-6, -4e-6
    depth = 8e-6

    pattern_settings = FibsemPatternSettings(
        pattern=FibsemPattern.Line,
        start_x=start_x,
        start_y=start_y,
        end_x=end_x,
        end_y=end_y,
        depth=depth,
    )
    milling_settings = FibsemMillingSettings(milling_current=5.6e-9, hfw=150e-6)

    # salami settings
    ss = SalamiSettings(n_steps=50, step_size=5e-9)

    salami.core.run_salami(microscope, settings, ss, pattern_settings, milling_settings)


def full_pipeline():
    

    # TODO: protocol
    # microscope, settings = utils.setup_session()
    
    # sa.run_salami_analysis(microscope, settings, "data")

    print("Hello Denoising Pipeline")

    # path = os.path.join(settings.image.save_path, cfg.DATA_DIR)
    # settings.image.save_path = path 
    # run_collection(microscope, settings)
    
    path = "/home/patrick/github/salami/demo_2023-03-02-06-32-14PM/data"
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
    run_segmentation(path)
    print("Saving results...")


def main():
    # run_salami_analysis("data")


    # fn()

    full_pipeline()


if __name__ == "__main__":
    main()