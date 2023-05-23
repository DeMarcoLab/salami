# # DENOISING
# model_filename = work_dir + '/models/model000499.pth'
# trainer = Trainer(work_dir, cfg.DN_DEPTH, cfg.DN_NN, device, batchsize=cfg.DN_BATCHSZ)
# trainer.load_model(model_filename)

# denoised_prefix = 'denoised_EB_'
# for index in range(50,76):
#     noisy_ref  =      img_prefix + index2str(index)   + img_fast_suffix # note that the ref for alignment has now fast dwell time
#     noisy_imgs =     [img_prefix + index2str(index-2) + img_fast_suffix]
#     noisy_imgs.append(img_prefix + index2str(index-1) + img_fast_suffix)
#     noisy_imgs.append(img_prefix + index2str(index)   + img_fast_suffix)
#     noisy_imgs.append(img_prefix + index2str(index+1) + img_fast_suffix)
#     noisy_imgs.append(img_prefix + index2str(index+2) + img_fast_suffix)
#     trainer.reconstruct(noisy_ref, noisy_imgs, denoised_prefix+index2str(index)+img_fast_suffix)
#########--------------------------------------------------#########

import datetime
import glob
import os
from pathlib import Path

from fibsem import utils

from salami import config as cfg
import math
from salami.denoise.trainer import Trainer


# # DENOISING
def load_model(
    checkpoint: Path,
    path: Path,
    device: str,
    depth: int = 5,
    nn: int = 5,
    batch_size: int = 500,
):
    trainer = Trainer(path, depth, nn, device, batchsize=batch_size)
    trainer.load_model(checkpoint)

    return trainer


def inference(trainer: Trainer, noisy_ref, noisy_imgs, fname: str, path: Path = None):
    trainer.reconstruct(
        noisy_ref, noisy_imgs, fname, path=path, pixelsize=10
    )  # TODO: pixelsize needs to be exposed?


# get a window of filenames around the index
def get_window(filenames, index, window_size=5):
    start = index - math.floor(window_size / 2)
    return filenames[start : start + window_size]


# get the index from the filename
def get_index(fname):
    return int(os.path.splitext(os.path.basename(fname))[0])


# get the index of the filename closest to the index
def get_closest_index(filenames, index):
    return min(filenames, key=lambda x: abs(get_index(x) - index))


def run_denoise_step(trainer: Trainer, fname: str, input_path: Path, output_path: Path):

    # get the current index to determine the window
    filenames = sorted(glob.glob(os.path.join(input_path, "*.tif")))
    idx = filenames.index(fname)
    
    # references for alignment
    noisy_ref = fname  # note that the ref for alignment has now fast dwell time
    noisy_imgs = get_window(filenames, idx, window_size=trainer.NN)

    if len(noisy_imgs) < trainer.NN:
        return
    inference(trainer, noisy_ref, noisy_imgs, path=output_path, fname=fname)


def setup_denoise_inference(conf: dict) -> Trainer:
    # working directory
    work_dir = f'denoising_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(work_dir, exist_ok=False)

    device = conf["device"]
    batch_size = int(conf["batch_size"])
    depth = int(conf["depth"])
    nn = int(conf["nn"])
    checkpoint = conf["checkpoint"]

    # load model
    trainer = load_model(
        checkpoint,
        path=work_dir,
        device=device,
        depth=depth,
        nn=nn,
        batch_size=batch_size,
    )

    return trainer


def run_denoise(input_path: Path, denoised_path: Path, conf: dict):
    trainer = setup_denoise_inference(conf)
    filenames = sorted(glob.glob(os.path.join(input_path, "*.tif")))

    for fname in filenames:
        run_denoise_step(
            trainer=trainer,
            fname=fname,
            input_path=input_path,
            output_path=denoised_path,
        )


def main():
    # input directory / path
    input_path = "/home/patrick/github/salami/data/20230310/raw/"

    # output directory / path
    output_path = "/home/patrick/github/salami/salami/output/"
    denoise_path = os.path.join(output_path, cfg.DENOISE_DIR)
    os.makedirs(denoise_path, exist_ok=True)

    # load protocol
    protocol = utils.load_yaml(
        "/home/patrick/github/salami/salami/protocol/protocol.yaml"
    )

    conf = protocol["denoise"]

    run_denoise(input_path, denoise_path, conf)


if __name__ == "__main__":
    main()


# TODO:
# - need to wait for n images before starting inference
# - need to implement the watchdog
# - need to implement the output directory structure
# - need to implement the collection directory structure

# ? do we need the reference just to do inference? investigate

# collection file structure
# raw/
#   fast/
#       00000.tif
#       00001.tif
#   ref/
#       00000.tif
#       00001.tif
# denoise/
#  00000.tif
#  00001.tif
# seg
#  00000.tif
#  00001.tif
