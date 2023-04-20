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



import numpy as np

def calculate_halfmap_frc(image):

    ny, nx = image.shape

    # # HALF SPLIT

    # # Split the image into two halves along the vertical axis
    # half1 = image[:nx//2, :nx//2]
    # half2 = image[:nx//2, nx//2:]

    # ny, nx = half1.shape

    ##### CHECKERBOARD SPLIT

    # create checkerboard mask 1px wide
    # mask = np.zeros_like(image, dtype=bool)
    # mask[::2, ::2] = 1
    # mask[1::2, 1::2] = 1

    # # split image with checkerboard mask
    # half1 = image * (mask == 1)
    # half2 = image * (mask == 0)

    ##### Double Checkerboard Split
    mask = np.zeros_like(image, dtype=bool)
    mask[::2, ::2] = 1
    mask[1::2, 1::2] = 1

    mask = mask.astype(int)

    idx = [x for x in range(0, mask.shape[0]) if x % 2 == 1]

    for i in idx:
        mask[i, :] = 2 * mask[i,  :]

    mask2 = mask == 0
    mask2 = mask2.astype(int)

    idx = [x for x in range(0, mask2.shape[0]) if x % 2 == 1]

    for i in idx:
        mask2[i, :] = 2 * mask2[i,  :]

    m00 = mask == 1
    m01 = mask == 2

    m10 = mask2 == 1
    m11 = mask2 == 2

    half00 = image * m00
    half01 = image * m01

    half10 = image * m10
    half11 = image * m11

    # return list[(img1, img2)]

    frcs = []
    for half1, half2 in [(half00, half01), (half10, half11)]:
            
        ny, nx = half1.shape

        half1 = (half1 - np.mean(half1) / np.std(half1))
        half2 = (half2 - np.mean(half2) / np.std(half2))

        # Perform Fourier transformation on the two halves
        half1_ft = np.fft.fftshift(np.fft.fft2(half1))
        half2_ft = np.fft.fftshift(np.fft.fft2(half2))

        # Calculate the dimensions and center of the Fourier transforms
        center = np.asarray(half1.shape) // (2 * 2 * np.sqrt(2)) # /2 for radisu / 2 for nyquist / sqrt(2) for diagonal
        
        # Initialize an array to store the Fourier ring correlation coefficients
        frc = np.zeros(int(min(center)))
        
        # Iterate through concentric rings from the center to the edge
        for r in range(frc.shape[0]):
            
            # Create a circular mask for the current ring
            mask = np.zeros(half1.shape, dtype=bool)
            
            # create meshgrid
            y, x = np.meshgrid(np.arange(-nx//2, nx//2), 
                            np.arange(-ny//2, ny//2))
            mask = np.round(np.sqrt(x**2 + y**2)) == r

            # Calculate the Fourier amplitudes within the current ring for both halves
            half1_ring = half1_ft[mask]
            half2_ring = half2_ft[mask]
        
            # Calculate the cross-correlation of the Fourier amplitudes in the current ring
            n1 = np.sum(half1_ring * np.conj(half2_ring))
            # v1 = np.sum(half1_ring * np.conj(half1_ring))
            # v2 = np.sum(half2_ring * np.conj(half2_ring))
            v1 = np.sum(np.abs(half1_ring) ** 2)
            v2 = np.sum(np.abs(half2_ring) ** 2)
            frc[r] = np.real(n1/ np.sqrt(v1*v2))

        frcs.append(frc)

    return frcs, (half00, half01, half10, half11)

def get_frc_mean(metric: np.ndarray) -> np.ndarray:
    return np.mean(metric, axis=0)

# # Example usage:
# # Load the image as a numpy array
# image = np.load("image.npy")

# # Call the function to calculate the half-map FRC
# frc = calculate_halfmap_frc(image)

# # Print the results
# print("Half-Map Fourier Ring Correlation:")
# print(frc)
