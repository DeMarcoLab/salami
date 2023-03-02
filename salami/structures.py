import glob
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tff
import zarr
from fibsem import acquire, alignment, calibration, milling
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (BeamType, FibsemMillingSettings, FibsemPattern,
                               FibsemPatternSettings, MicroscopeSettings)


@dataclass
class SalamiSettings:
    n_steps: int
    step_size: float
    _align: bool = True
    _milling: bool = True
    _neutralise: bool = True


def run_salami(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    salami_settings: SalamiSettings,
    pattern_settings: FibsemPatternSettings,
    milling_settings: FibsemMillingSettings,
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

            # create pattern
            milling.setup_milling(microscope)
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
        settings.image.label = f"{i:04d}"
        eb_image = acquire.new_image(microscope, settings.image)

        # update pattern
        pattern_settings.start_y += step_size
        pattern_settings.end_y += step_size

        # # manually adjust working distance
        # wd_diff = step_size * np.sin(np.deg2rad(38))
        # microscope.beams.electron_beam.working_distance.value -= wd_diff #4e-3# 3.995e-3

        # if i % 50 == 0:
        # microscope.autocontrast(BeamType.ELECTRON)


def create_sweep_parameters(settings: MicroscopeSettings, conf: dict = None):

    # pixelsize (nm), # pixelsize = hfw/n_pixels_x
    pixelsizes = np.array([1, 2, 3, 4, 5, 6, 8, 10])  # , 12, 14, 16, 18, 20])
    resolutions = [[1536, 1024], [3072, 2048]]  # , [6144, 4096]]
    voltages = [1e3, 2e3]
    currents = [1.6e-9, 0.1e-9, 0.2e-9]  # , 0.4e-9, 0.8e-9]
    dwell_times = [0.5e-6, 1e-6, 3e-6]  # , 5e-6, 8e-6, 20e-6]

    base_path = os.path.join(settings.image.save_path, "data")

    df = pd.DataFrame(
        columns=[
            "voltage",
            "current",
            "resolution_x",
            "resolution_y",
            "hfw",
            "dwell_time",
        ]
    )
    parameters = []
    for voltage in voltages:
        for current in currents:
            for resolution in resolutions:
                hfws = pixelsizes * resolution[0] * 1e-9  # nm
                for hfw in hfws:
                    for dwell_time in dwell_times:

                        idx = len(parameters)
                        idx = f"{idx:04d}"
                        data_path = os.path.join(base_path, idx)
                        os.makedirs(data_path, exist_ok=True)

                        params = {
                            "voltage": voltage,
                            "current": current,
                            "resolution_x": resolution[0],
                            "resolution_y": resolution[1],
                            "hfw": hfw,
                            "dwell_time": dwell_time,
                            "idx": idx,
                            "path": data_path,
                        }
                        parameters.append(params)

    logging.info(f"{len(parameters)} Sweeps to be performed")
    df = pd.DataFrame(parameters)
    df.to_csv(os.path.join(base_path, "parameters.csv"), index=False)

    return df


def run_sweep_collection(microscope, settings, conf: dict = None, break_idx: int = 10):

    base_path = os.path.join(settings.image.save_path, "data")

    df = pd.read_csv(os.path.join(base_path, "parameters.csv"))
    params_dict = df.to_dict("records")

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
    ss = SalamiSettings(n_steps=5, step_size=5e-9)

    # sweep through all params, collect data
    for i, parameters in enumerate(params_dict):

        voltage = float(parameters["voltage"])
        current = float(parameters["current"])
        resolution = [int(parameters["resolution_x"]), int(parameters["resolution_y"])]
        hfw = float(parameters["hfw"])
        dwell_time = float(parameters["dwell_time"])
        idx = parameters["idx"]
        data_path = parameters["path"]

        logging.info(
            f"SWEEP ({i}/{len(params_dict)}) -- voltage: {voltage}, current: {current}, resolution: {resolution}, hfw: {hfw:.2e}, dwell_time: {dwell_time}"
        )

        # set microscope params
        microscope.set("voltage", voltage, BeamType.ELECTRON)
        microscope.set("current", current, BeamType.ELECTRON)

        # set imaging params
        settings.image.resolution = resolution
        settings.image.dwell_time = dwell_time
        settings.image.hfw = hfw
        settings.image.save_path = data_path

        # run salami
        run_salami(microscope, settings, ss, pattern_settings, milling_settings)

        if i == break_idx:
            break


def run_sweep_analysis(path: Path, conf: dict = None):

    df = pd.read_csv(os.path.join(path, "parameters.csv"))

    params_dict = df.to_dict("records")

    def calc_metric(images):
        # TODO: replace with frsc
        # calculate metric
        metric = da.average(images).compute()
        return metric + np.random.rand() * 50

    # FSC:
    # - https://en.wikipedia.org/wiki/Fourier_shell_correlation
    # - https://www.nature.com/articles/s41467-019-11024-z

    # PolishEM:
    # - https://academic.oup.com/bioinformatics/article/36/12/3947/5813331?login=true
    # - https://sites.google.com/site/3demimageprocessing/polishem



    path_data = []
    for i, parameters in enumerate(params_dict):
        data_path = parameters["path"]

        metric = 0
        n_images = 0

        # get all images
        filenames = sorted(glob.glob(os.path.join(data_path, "*.tif")))
        if len(filenames) > 0:
            images = da.from_zarr(
                tff.imread(os.path.join(data_path, "*.tif*"), aszarr=True)
            )
            metric = calc_metric(images)
            n_images = images.shape[0]

        path_data.append([data_path, n_images, metric])

    df = pd.DataFrame(path_data, columns=["path", "n_images", "metric"])
    df.to_csv(os.path.join(path, "metrics.csv"), index=False)

    return df


def join_df(path: Path, conf: dict = None):
    # join parameters and metrics dataframes
    df = pd.read_csv(os.path.join(path, "parameters.csv"))
    df_metrics = pd.read_csv(os.path.join(path, "metrics.csv"))

    df = df.join(df_metrics.set_index("path"), on="path")
    df.to_csv(os.path.join(path, "parameters_metrics.csv"), index=False)

    return df


def plot_metrics(path: Path, conf: dict = None):
    # plot metrics
    df = pd.read_csv(os.path.join(path, "parameters_metrics.csv"))

    # drop rows with no images
    df = df[df["n_images"] > 0]
    df = df.sort_values(by="metric", ascending=False)

    fig, ax = plt.subplots(3, 2, figsize=(7, 7))

    fig.suptitle("Sweep Metrics (Work in Progress)")

    # plot each column against the metric, except the path and idx column
    for i, col in enumerate(df.columns):
        if col not in ["path", "idx", "n_images", "metric"]:
            ax[i // 2, i % 2].scatter(df[col], df["metric"])

            # add title and axis labels
            ax[i // 2, i % 2].set_title(f"{col} vs metric")
            ax[i // 2, i % 2].set_xlabel(col)
            ax[i // 2, i % 2].set_ylabel("metric")

    plt.tight_layout()
    plt.show()
