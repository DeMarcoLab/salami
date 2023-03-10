

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
from salami.denoise.align import *
from salami.denoise.trainer import Trainer
# from salami.denoise import config as cfg 
import tqdm
import os
from pathlib import Path

from pprint import pprint

import datetime


# # utility function to handle filenaming convention, custom per dataset
# def index2str(i):
#     # if i < 1000:
#     #     str_index = '%03d' % i
#     # else:
#     #     str_index = '%d' % i
#     str_index = "%d" % i
#     return str_index



# # DENOISING
def load_model(checkpoint:Path, path: Path, device: str, depth: int = 5, nn: int = 5, batch_size: int = 500):

    trainer = Trainer(path, depth, nn, device, batchsize=batch_size)
    trainer.load_model(checkpoint)

    return trainer

def inference(trainer: Trainer, noisy_ref, noisy_imgs, fname: str, path:Path = None):
    trainer.reconstruct(noisy_ref, noisy_imgs, fname, path=path)


# def get_image_stack(index, path:Path, prefix:str, suffix: str, num_imgs=5):
#     start = index - math.floor(num_imgs / 2)
#     return [get_denoised_fname(start+i, path, prefix, suffix) for i in range(num_imgs)]

# def get_denoised_fname(index, path:Path, prefix:str, suffix: str):
#     return os.path.join(path, f"{prefix}{index2str(index)}{suffix}")


# get a window of filenames around the index
def get_window(filenames, index, window_size=5):
    start = index - math.floor(window_size / 2)
    return filenames[start:start+window_size]

# get the index from the filename
def get_index(fname):
    return int(os.path.splitext(os.path.basename(fname))[0])


# get the index of the filename closest to the index
def get_closest_index(filenames, index):
    return min(filenames, key=lambda x: abs(get_index(x) - index))

def run_denoise(input_path: Path, denoised_path:Path, conf:dict):

    work_dir = conf["work_dir"] # TODO: rm this directory as it is only meant to be temp?
    # img_prefix = conf["img_prefix"] # TODO: come up with a better scheme than the image prefix
    # img_fast_suffix = conf["img_fast_suffix"]
    device = conf["device"]
    batch_size = conf["batch_size"]
    depth = conf["depth"]
    nn = conf["nn"]
    checkpoint = conf["checkpoint"]

    # load model
    trainer = load_model(checkpoint, path=work_dir, 
                            device=device, depth=depth, 
                            nn=nn, batch_size=batch_size)

    # for index in range(50,76): # how to generalise
    #     noisy_ref  =  get_denoised_fname(index, input_path, img_prefix, img_fast_suffix) # note that the ref for alignment has now fast dwell time   
    #     noisy_imgs = get_image_stack(index, path=input_path, prefix=img_prefix, suffix=img_fast_suffix, num_imgs=nn)
    #     fname = f"denoise_{index:06d}.tif"
        
    #     pprint(noisy_imgs)
    #     print("filename:", fname)
    #     inference(trainer, noisy_ref, noisy_imgs, path=denoised_path, fname=fname)

    import glob

    filenames = sorted(glob.glob(os.path.join(input_path, "fast", "*.tif")))

    for idx, fname in enumerate(filenames):
        noisy_ref  =  fname # note that the ref for alignment has now fast dwell time
        noisy_imgs = get_window(filenames, idx, window_size=nn)

        # print("filename:", fname)
        # pprint(noisy_imgs)
        if len(noisy_imgs) < nn:
            continue
        inference(trainer, noisy_ref, noisy_imgs, path=denoised_path, fname=fname)

        

# img_prefix = "EB_"
# img_fast_suffix = "_5e-07_0_eb.tif"

# CUDA device
# device = "cuda:0"
# batch_size = 500 # cfg.DN_BATCHSZ
# depth = 5 # cfg.DN_DEPTH
# nn = 5 # cfg.DN_NN
# checkpoint = "/home/patrick/github/salami/salami/models/model000499.pth"


from fibsem import utils
from salami import config as cfg 
def main():

    # working directory
    work_dir = f'denoising_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(work_dir, exist_ok=False)

    # input directory / path
    # input_path = "/home/patrick/github/salami/salami/denoise/raw_imgs/"
    input_path = "/home/patrick/github/salami/data/20230310/raw/"

    # output directory / path
    output_path = "/home/patrick/github/salami/salami/output/"
    denoise_path = os.path.join(output_path, cfg.DENOISE_DIR)
    os.makedirs(denoise_path, exist_ok=True)

    # load protocol
    protocol = utils.load_yaml("/home/patrick/github/salami/salami/protocol/protocol.yaml")

    conf = protocol["denoise"]
    conf["work_dir"] = work_dir

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