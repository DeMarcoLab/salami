import glob
import logging
import os
from pathlib import Path

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tff
import zarr
from fibsem.structures import (
    BeamType,
    FibsemMillingSettings,
    FibsemPattern,
    FibsemPatternSettings,
    MicroscopeSettings,
)

from salami.core import run_salami
from salami.structures import SalamiSettings


def create_sweep_parameters(settings: MicroscopeSettings, conf: dict = None):

    # pixelsize (nm), # pixelsize = hfw/n_pixels_x
    pixelsizes = np.array(conf["pixelsize"])  # nm
    resolutions = conf["resolution"]
    voltages = conf["voltage"]
    currents = conf["current"]
    dwell_times = conf["dwell_time"]

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
    ss = SalamiSettings(n_steps=50, step_size=5e-9)

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


def run_salami_analysis(microscope, settings, path: Path):

    path = os.path.join(settings.image.save_path, "data")

    from fibsem import utils
    from salami import config as cfg

    conf = utils.load_yaml(cfg.SWEEP_PATH)

    df = create_sweep_parameters(settings, conf)

    run_sweep_collection(microscope, settings, break_idx=1)

    df = run_sweep_analysis(path)

    df = join_df(path)

    plot_metrics(path)
