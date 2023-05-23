import logging
import time
from fibsem import acquire, alignment, calibration, milling
from fibsem.microscope import FibsemMicroscope
from fibsem.patterning import FibsemMillingStage, get_pattern
from fibsem.structures import (BeamSettings, BeamType, FibsemDetectorSettings,
                               FibsemMillingSettings, FibsemPattern,
                               FibsemPatternSettings, ImageSettings,
                               MicroscopeSettings)

from salami.structures import SalamiImageSettings, SalamiSettings
import os


def _update_ui(parent_ui, info:dict):

    if parent_ui is not None:
        parent_ui.update_signal.emit(info)


def run_salami(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    salami_settings: SalamiSettings,
    parent_ui=None,
):

    n_steps = salami_settings.n_steps
    step_size = salami_settings.step_size

    base_path = salami_settings.image[0].image.save_path # TOOD: need to get the experiment path from somehwere?

    eb_image = None

    for i in range(0, n_steps):

        info = {"name": "Salami", "step": i, "total_steps": n_steps}

        logging.info(
            f" -------------------------------- SLICE {i}/{n_steps} -------------------------------- "
        )

        # slice
        MILL_START_IDX = 0
        if i > MILL_START_IDX:
            
            info["name"] = "Milling"
            _update_ui(parent_ui, info)
            
            # define pattern
            salami_settings.mill.pattern.define(protocol=salami_settings.mill.pattern.protocol, point=salami_settings.mill.pattern.point) 
            milling.mill_stages(microscope=microscope, settings=settings, stages=[salami_settings.mill])            


        # neutralise charge
        if salami_settings._neutralise:
            info["name"] = "Neutralising Charge"
            _update_ui(parent_ui, info)
            settings.image.save = False
            calibration.auto_charge_neutralisation(
                microscope, settings.image, n_iterations=5
            )

        # align
        # TODO: this needs to happen for each image?
        if salami_settings._align and eb_image is not None:
            info["name"] = "Aligning"
            _update_ui(parent_ui, info)
            alignment.beam_shift_alignment(microscope, settings.image, eb_image)

        # view
        # acquire

        for j, img_settings in enumerate(salami_settings.image):
            info["name"] =  f"Imaging {j+1}/{len(salami_settings.image)}"
            _update_ui(parent_ui ,info)

            img_settings.image.save_path = os.path.join(base_path, f"{j:03d}")
            os.makedirs(img_settings.image.save_path, exist_ok=True)

            # TODO: make this more conveinent generally
            # set beam settings
            microscope.set("working_distance", img_settings.beam.working_distance, BeamType.ELECTRON)
            microscope.set("current", img_settings.beam.beam_current, BeamType.ELECTRON)
            microscope.set("stigmation", img_settings.beam.stigmation, BeamType.ELECTRON)
            microscope.set("shift", img_settings.beam.shift, BeamType.ELECTRON)

            # set detector settings
            microscope.set("detector_type", img_settings.detector.type, BeamType.ELECTRON)
            microscope.set("detector_mode", img_settings.detector.mode, BeamType.ELECTRON)

            # set image settings
            img_settings.image.save = True
            img_settings.image.autocontrast = False
            img_settings.image.beam_type = BeamType.ELECTRON
            img_settings.image.label = f"{i:06d}"
            eb_image = acquire.new_image(microscope, img_settings.image)
            
            info["name"] = "Image Update"
            info["image"] = eb_image
            _update_ui(parent_ui, info)
            time.sleep(1)


        # update pattern
        # increase point.y by step_size
        salami_settings.mill.pattern.point.y += step_size

        # # manually adjust working distance
        # wd_diff = step_size * np.sin(np.deg2rad(38))
        # microscope.beams.electron_beam.working_distance.value -= wd_diff #4e-3# 3.995e-3

        # if i % 50 == 0:
        # microscope.autocontrast(BeamType.ELECTRON)


        info["name"] = "Done"
        info["milling_stage"] = salami_settings.mill
        _update_ui(parent_ui, info)


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
            FibsemDetectorSettings()  # TODO: add these to protocol
            )
        ],
        mill = FibsemMillingStage(
            milling=ms,
            pattern=ps
        )
    )


    return ss