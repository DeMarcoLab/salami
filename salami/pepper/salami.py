import logging
from dataclasses import dataclass

import numpy as np
from fibsem import acquire, alignment, calibration, milling
from fibsem.structures import BeamType

# TODO: move this to salami project
@dataclass
class SalamiSettings:
    n_steps: int
    step_size: float
    _align: bool = True
    _milling: bool = True
    _neutralise: bool = True


def run_salami(microscope, settings, salami_settings: SalamiSettings, pattern_settings, milling_settings):

    n_steps = salami_settings.n_steps
    step_size = salami_settings.step_size

    eb_image = None

    for i in range(0, n_steps):

        logging.info(f" ---------------- SLICE {i}/{n_steps} ---------------- ")
        
        # slice
        MILL_START_IDX = 0
        if i > MILL_START_IDX:

            # create pattern
            milling.setup_milling(microscope)
            milling.draw_line(microscope, pattern_settings=pattern_settings)

            # run
            milling.run_milling(microscope, milling_current=milling_settings.milling_current)
            milling.finish_milling(microscope, imaging_current=settings.system.ion.current)
        
        # neutralise charge
        if salami_settings._neutralise:
            settings.image.save = False
            calibration.auto_charge_neutralisation(microscope, settings.image, n_iterations=5)

        # align
        if salami_settings._align and eb_image is not None:
            alignment.beam_shift_alignment(microscope, settings.image, eb_image)

        # view
        # acquire
        settings.image.save = True
        settings.image.autocontrast = False
        settings.image.beam_type = BeamType.ELECTRON
        settings.image.label = f"{i:04d}"
        eb_image = acquire.new_image(microscope, settings.image)
        
        # update pattern
        pattern_settings.start_y += step_size
        pattern_settings.end_y  += step_size
        
        # # manually adjust working distance
        # wd_diff = step_size * np.sin(np.deg2rad(38))
        # microscope.beams.electron_beam.working_distance.value -= wd_diff #4e-3# 3.995e-3 

        # if i % 50 == 0:
            # microscope.autocontrast(BeamType.ELECTRON)

