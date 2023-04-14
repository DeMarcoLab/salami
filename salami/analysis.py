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
from fibsem import utils, acquire
from fibsem.structures import (BeamType, FibsemImage, FibsemMillingSettings,
                               FibsemPattern, FibsemPatternSettings,
                               MicroscopeSettings)

from salami import config as cfg
from salami.core import load_protocol, run_salami
from salami.structures import SalamiSettings


def run_salami_analysis(path: Path = None, break_idx: int = 10):

    microscope, settings = utils.setup_session(protocol_path=cfg.PROTOCOL_PATH)

    if path is None:
        path = os.path.join(settings.image.save_path, "data")
    
    os.makedirs(path, exist_ok=True)
    
    ss = load_protocol(settings.protocol)

    # load sweep parameters
    conf = utils.load_yaml(cfg.SWEEP_PATH)
    create_sweep_parameters(settings, conf, path=path)

    # run sweep
    run_sweep_collection(microscope, settings, ss=ss, break_idx=break_idx, path=path)

    run_sweep_analysis(path)

    plot_metrics(path)


def create_sweep_parameters(settings: MicroscopeSettings, conf: dict = None, path: Path = None):

    # pixelsize (nm), # pixelsize = hfw/n_pixels_x
    pixelsizes = np.array(conf["pixelsize"])  # nm
    resolutions = conf["resolution"]
    voltages = conf["voltage"]
    currents = conf["current"]
    dwell_times = conf["dwell_time"]

    if path is None:
        path = os.path.join(settings.image.save_path, "data")

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
                        idx = f"{idx:06d}"
                        # data_path = os.path.join(base_path, idx)
                        os.makedirs(path, exist_ok=True)

                        params = {
                            "voltage": voltage,
                            "current": current,
                            "resolution_x": resolution[0],
                            "resolution_y": resolution[1],
                            "hfw": hfw,
                            "pixelsize": hfw / resolution[0] * 1e9, # nm
                            "dwell_time": dwell_time,
                            "idx": idx,
                            "path": path,
                        }
                        parameters.append(params)

    logging.info(f"{len(parameters)} Sweeps to be performed")
    df = pd.DataFrame(parameters)
    df.to_csv(os.path.join(path, "parameters.csv"), index=False)

    return df


def run_sweep_collection(microscope, settings, ss: SalamiSettings, conf: dict = None, break_idx: int = 10, path: Path = None):

    if path is None:
        path = os.path.join(settings.image.save_path, "data")

    df = pd.read_csv(os.path.join(path, "parameters.csv"))
    params_dict = df.to_dict("records")

    # image settings
    settings.image.save = True
    settings.image.autocontrast = False
    settings.image.gamma_enabled = False


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
        settings.image.label = idx = f"{idx:06d}"
        settings.image.save = True
        settings.image.autocontrast = False
        settings.image.gamma_enabled = False

        # run salami
        # run_salami(microscope, settings, ss)
        acquire.new_image(microscope, settings.image)

        if i == break_idx:
            break


def run_sweep_analysis(path: Path, conf: dict = None):

    df = pd.read_csv(os.path.join(path, "parameters.csv"))

    params_dict = df.to_dict("records")

    # def calcFRSC(images):
    #     # TODO: replace with frsc
    #     # calculate metric
    #     metric = da.average(images).compute()
    #     return metric + np.random.rand() * 50

    calc_metric = calcFRC

    # FSC:
    # - https://en.wikipedia.org/wiki/Fourier_shell_correlation
    # - https://www.nature.com/articles/s41467-019-11024-z

    # PolishEM:
    # - https://academic.oup.com/bioinformatics/article/36/12/3947/5813331?login=true
    # - https://sites.google.com/site/3demimageprocessing/polishem

    path_data = []
    for i, parameters in enumerate(params_dict):
        data_path = parameters["path"]
        idx = parameters["idx"]

        fname = os.path.join(data_path, f"{idx:06d}_eb.tif")

        # print(parameters)

        img = FibsemImage.load(fname)

        # img1, img2 = np.split(img, 2, axis=1)
        # metric = calc_metric(img1, img2)
        metric = np.mean(img.data)

        path_data.append([idx, metric])

        print(f"Path: {os.path.basename(fname)}, Metric: {metric:.2f}, ")
        print("-"*50)


    # save metrics
    df = pd.DataFrame(path_data, columns=["idx", "metric"])
    df.to_csv(os.path.join(path, "metrics.csv"), index=False)

    # join parameters and metrics dataframes
    df = join_df(path)

    return df

    #     metric = 0
    #     n_images = 0

    #     # get all images
    #     filenames = sorted(glob.glob(os.path.join(data_path, "*.tif")))
    #     if len(filenames) > 0:
    #         images = da.from_zarr(
    #             tff.imread(os.path.join(data_path, "*.tif*"), aszarr=True)
    #         )

    #         # split image into two halves
    #         metrics = []
    #         for img in images:
    #             img1, img2 = np.split(img, 2, axis=1)
    #             metrics.append(calc_metric(img1, img2))
    #         # metrics = [calc_metric(img) for img in images]
    #         n_images = images.shape[0]
    #         metric = np.mean(metrics)

    #     path_data.append([data_path, n_images, metric])

    # # save metrics
    # df = pd.DataFrame(path_data, columns=["path", "n_images", "metric"])
    # df.to_csv(os.path.join(path, "metrics.csv"), index=False)

    # # join parameters and metrics dataframes
    # df = join_df(path)

    # return df


def join_df(path: Path, conf: dict = None):
    # join parameters and metrics dataframes
    df = pd.read_csv(os.path.join(path, "parameters.csv"))
    df_metrics = pd.read_csv(os.path.join(path, "metrics.csv"))

    df = df.join(df_metrics.set_index("idx"), on="idx")
    df.to_csv(os.path.join(path, "parameters_metrics.csv"), index=False)

    return df


def plot_metrics(path: Path, conf: dict = None):
    # plot metrics
    df = pd.read_csv(os.path.join(path, "parameters_metrics.csv"))

    # plot 3d scatter plot, metric vs current vs pixelsize
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df["current"], df["pixelsize"], df["metric"], c=df["hfw"])
    ax.set_xlabel("Current")
    ax.set_ylabel("Pixelsize")
    ax.set_zlabel("Metric")
    
    ax.legend(loc="best")

    plt.show()




# def plot_metrics(path: Path, conf: dict = None):
#     # plot metrics
#     df = pd.read_csv(os.path.join(path, "parameters_metrics.csv"))

#     # drop rows with no images
#     df = df[df["n_images"] > 0]
#     df = df.sort_values(by="metric", ascending=False)

#     fig, ax = plt.subplots(3, 2, figsize=(7, 7))

#     fig.suptitle("Sweep Metrics (Work in Progress)")

#     # plot each column against the metric, except the path and idx column
#     for i, col in enumerate(df.columns):
#         if col not in ["path", "idx", "n_images", "metric"]:
#             ax[i // 2, i % 2].scatter(df[col], df["metric"])

#             # add title and axis labels
#             ax[i // 2, i % 2].set_title(f"{col} vs metric")
#             ax[i // 2, i % 2].set_xlabel(col)
#             ax[i // 2, i % 2].set_ylabel("metric")

#     plt.tight_layout()
#     plt.show()





# this is only for one image?
# see frame.PSNR for two images?
def calcPSNR(img: FibsemImage, max_val: float = 255.0) -> float:
    # https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    data = img.data
    mse = np.mean((data - data.mean()) ** 2)
    return 10 * np.log10(max_val ** 2 / mse)

def calcFRC(img1: FibsemImage, img2: FibsemImage) -> float:
    # https://en.wikipedia.org/wiki/Fourier_ring_correlation
    # https://www.nature.com/articles/s41467-019-11024-z
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3885820/

    from fibsem.imaging.utils import normalise_image

    # normalise images
    img1 = normalise_image(img1)
    img2 = normalise_image(img2)

    # apply tukey window
    # https://en.wikipedia.org/wiki/Tukey_window
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.tukey.html

    import scipy.signal as signal
    window = signal.tukey(img1.shape[0], alpha=0.9)
    img1 = img1 * window[:, np.newaxis]
    img2 = img2 * window[:, np.newaxis]
    

    # calculate the 2D FFT
    f1 = np.fft.fft2(img1)
    f2 = np.fft.fft2(img2)

    # calculate the FRC
    frc = np.abs(f1 * f2.conj()) / np.sqrt(np.abs(f1 * f1.conj()) * np.abs(f2 * f2.conj()))

    # calculate the radial average
    frc = calc_radial_average(frc)

    # calculate the FRC
    frc = frc / frc[0]

    return frc
    

def calc_radial_average(img: np.ndarray) -> float:
    # https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile

    y, x = np.indices((img.shape))
    r = np.sqrt((x - x.mean()) ** 2 + (y - y.mean()) ** 2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), img.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr

    return radialprofile