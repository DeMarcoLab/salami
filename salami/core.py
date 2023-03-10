import logging

from fibsem import acquire, alignment, calibration, milling
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    BeamType,
    FibsemMillingSettings,
    FibsemPattern,
    FibsemPatternSettings,
    MicroscopeSettings,
)
from salami.structures import SalamiSettings


def run_salami(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    salami_settings: SalamiSettings,
):

    # image settings
    settings.image.save = True
    settings.image.autocontrast = False
    settings.image.gamma_enabled = False

    pattern_settings = salami_settings.pattern
    milling_settings = salami_settings.milling
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

            # create pattern
            milling.setup_milling(microscope, milling_settings)
            milling.draw_line(microscope, pattern_settings=pattern_settings)

            # run
            milling.run_milling(
                microscope, milling_current=milling_settings.milling_current
            )
            milling.finish_milling(
                microscope, imaging_current=settings.system.ion.current
            )

        # neutralise charge
        if salami_settings._neutralise:
            settings.image.save = False
            calibration.auto_charge_neutralisation(
                microscope, settings.image, n_iterations=5
            )

        # align
        if salami_settings._align and eb_image is not None:
            alignment.beam_shift_alignment(microscope, settings.image, eb_image)

        # view
        # acquire
        settings.image.save = True
        settings.image.autocontrast = False
        settings.image.beam_type = BeamType.ELECTRON
        settings.image.label = f"{i:06d}"
        eb_image = acquire.new_image(microscope, settings.image)

        # update pattern
        if pattern_settings.pattern is FibsemPattern.Line:
            pattern_settings.start_y += step_size
            pattern_settings.end_y += step_size
        elif pattern_settings.pattern is FibsemPattern.Rectangle:
            pattern_settings.centre_y += step_size

        # # manually adjust working distance
        # wd_diff = step_size * np.sin(np.deg2rad(38))
        # microscope.beams.electron_beam.working_distance.value -= wd_diff #4e-3# 3.995e-3

        # if i % 50 == 0:
        # microscope.autocontrast(BeamType.ELECTRON)


def load_protocol(protocol: dict) -> SalamiSettings:

    # pattern settings
    ps = FibsemPatternSettings.__from_dict__(protocol["milling"]["pattern"])

    ms = FibsemMillingSettings.__from_dict__(protocol["milling"])

    # salami settings
    ss = SalamiSettings(
        n_steps=int(protocol["num_steps"]),
        step_size=float(protocol["step_size"]),
        pattern=ps,
        milling=ms,
    )

    return ss