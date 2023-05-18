import logging

from fibsem import acquire, alignment, calibration, milling
from fibsem.microscope import FibsemMicroscope
from fibsem.patterning import FibsemMillingStage, get_pattern
from fibsem.structures import (BeamSettings, BeamType, FibsemDetectorSettings,
                               FibsemMillingSettings, FibsemPattern,
                               FibsemPatternSettings, ImageSettings,
                               MicroscopeSettings)

from salami.structures import SalamiImageSettings, SalamiSettings


def run_salami(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    salami_settings: SalamiSettings,
):

    n_steps = salami_settings.n_steps
    step_size = salami_settings.step_size

    eb_image = None

    for i in range(0, n_steps):

        logging.info(
            f" -------------------------------- SLICE {i}/{n_steps} -------------------------------- "
        )

        # slice
        MILL_START_IDX = 0
        if i > MILL_START_IDX:
            
            # define pattern
            salami_settings.mill.pattern.define(protocol=salami_settings.mill.pattern.protocol, point=salami_settings.mill.pattern.point)

            milling.mill_stages(microscope=microscope, settings=settings, stages=[salami_settings.mill])            


        # neutralise charge
        if salami_settings._neutralise:
            settings.image.save = False
            calibration.auto_charge_neutralisation(
                microscope, settings.image, n_iterations=5
            )

        # align
        # TODO: this needs to happen for each image?
        if salami_settings._align and eb_image is not None:
            alignment.beam_shift_alignment(microscope, settings.image, eb_image)

        # view
        # acquire

        for img_settings in salami_settings.image:
            img_settings.image.save = True
            img_settings.image.autocontrast = False
            img_settings.image.beam_type = BeamType.ELECTRON
            img_settings.image.label = f"{i:06d}"
            eb_image = acquire.new_image(microscope, img_settings.image)


        # update pattern
        # increase point.y by step_size
        salami_settings.mill.pattern.point.y += step_size

        # # manually adjust working distance
        # wd_diff = step_size * np.sin(np.deg2rad(38))
        # microscope.beams.electron_beam.working_distance.value -= wd_diff #4e-3# 3.995e-3

        # if i % 50 == 0:
        # microscope.autocontrast(BeamType.ELECTRON)


def load_protocol(protocol: dict) -> SalamiSettings:

    # pattern settings
    ps = get_pattern(protocol["milling"]["pattern"]["name"]).__from_dict__(protocol["milling"]["pattern"])
    ms = FibsemMillingSettings.__from_dict__(protocol["milling"])


    beam_type = BeamType.ELECTRON

    ss = SalamiSettings(
        n_steps=int(protocol["num_steps"]),
        step_size=float(protocol["step_size"]),
        image = [SalamiImageSettings(
            ImageSettings.__from_dict__(protocol["imaging"]["image"]),
            BeamSettings(beam_type=beam_type), 
            FibsemDetectorSettings()
            )
        ],
        mill = FibsemMillingStage(
            milling=ms,
            pattern=ps
        )
    )


    return ss