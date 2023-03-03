from align import *
from trainer import Trainer
import config as cfg
import tqdm
import os

import datetime

# work directory, needs to exists
work_dir = f'denoising_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(work_dir, exist_ok=False)
raw_imgs_dir = "raw_imgs/"
img_prefix = raw_imgs_dir + "EB_"
img_slow_suffix = "_5e-06_0_eb.tif"
img_fast_suffix = "_5e-07_0_eb.tif"

# CUDA device
device = "cuda:0"

# utility function to handle filenaming convention, custom per dataset
def index2str(i):
    # if i < 1000:
    #     str_index = '%03d' % i
    # else:
    #     str_index = '%d' % i
    str_index = "%d" % i
    return str_index


#
# ------ 1. FULL PIPELINE PROTOCOL ------
#

# INIT, supervised training
aligner = Aligner(work_dir, cfg.PIXSZ, nthreads=8)
trainer = Trainer(
    work_dir,
    cfg.DN_DEPTH,
    cfg.DN_NN,
    device,
    batchsize=cfg.DN_BATCHSZ,
    dropout=0.3,
    lr=0.01,
    naugment=cfg.DN_NAUGMENT,
)

# ALIGNMENT & TILES GENERATIONS
index_list = range(50, 76, 5)
for index in tqdm.tqdm(index_list):
    noisy_ref = img_prefix + index2str(index) + img_slow_suffix
    noisy_imgs = [img_prefix + index2str(index - 2) + img_fast_suffix]
    noisy_imgs.append(img_prefix + index2str(index - 1) + img_fast_suffix)
    noisy_imgs.append(img_prefix + index2str(index) + img_fast_suffix)
    noisy_imgs.append(img_prefix + index2str(index + 1) + img_fast_suffix)
    noisy_imgs.append(img_prefix + index2str(index + 2) + img_fast_suffix)
    aligned_ref, aligned_imgs = aligner.align_images(
        noisy_ref, noisy_imgs, align_patch=cfg.AL_PATCH
    )
    train_ref, valid_ref, train_inp, valid_inp = aligner.generate_denoising_tiles(
        aligned_ref, aligned_imgs
    )
    trainer.train_dataset.push_stacks(train_ref, train_inp)
    trainer.valid_dataset.push_stacks(valid_ref, valid_inp)
    # aligned images are no longer necessary
    os.remove(aligned_ref)
    for img in aligned_imgs:
        os.remove(img)

# For resuming only
trainer.train_dataset.write_stacks_list(work_dir + "/train_stacks_list.csv")
trainer.valid_dataset.write_stacks_list(work_dir + "/valid_stacks_list.csv")

# TRAINING
trainer.init_dataloaders()
trainer.deactivate_optimizer_scheduler()

# reconstruct some image every 20 epochs for debugging purpose
index = 62
noisy_ref = img_prefix + index2str(index) + img_fast_suffix
noisy_imgs = [img_prefix + index2str(index - 2) + img_fast_suffix]
noisy_imgs.append(img_prefix + index2str(index - 1) + img_fast_suffix)
noisy_imgs.append(img_prefix + index2str(index) + img_fast_suffix)
noisy_imgs.append(img_prefix + index2str(index + 1) + img_fast_suffix)
noisy_imgs.append(img_prefix + index2str(index + 2) + img_fast_suffix)

# Actual training
lr = 0.01
lr_decay = 0.85
trainer.set_lr(lr)
for i in range(25):
    for j in range(20):
        trainer.optimize()
    lr *= lr_decay
    trainer.set_lr(lr)

    # reconstruct the same image every 20 epochs for debugging purpose
    trainer.reconstruct(
        noisy_ref, noisy_imgs, "test_img_" + str(i * 20 + j) + "_denoised.tif"
    )

trainer.optimize()  # just to get to epoch 500, which is commited to file
model_filename = trainer.get_current_model()

# DENOISING
denoised_prefix = "denoised_EB_"
for index in range(50, 76):
    noisy_ref = (
        img_prefix + index2str(index) + img_fast_suffix
    )  # note that the ref for alignment has now fast dwell time
    noisy_imgs = [img_prefix + index2str(index - 2) + img_fast_suffix]
    noisy_imgs.append(img_prefix + index2str(index - 1) + img_fast_suffix)
    noisy_imgs.append(img_prefix + index2str(index) + img_fast_suffix)
    noisy_imgs.append(img_prefix + index2str(index + 1) + img_fast_suffix)
    noisy_imgs.append(img_prefix + index2str(index + 2) + img_fast_suffix)
    trainer.reconstruct(
        noisy_ref, noisy_imgs, denoised_prefix + index2str(index) + img_fast_suffix
    )


# #
# # ------ 2. RESUME FROM ALIGNMENT & GENERATED TILES ------
# #

# # TRAINING
# trainer = Trainer(work_dir, cfg.DN_DEPTH, cfg.DN_NN, device, batchsize=cfg.DN_BATCHSZ, dropout=0.3, lr=0.01, naugment=cfg.DN_NAUGMENT)
# # import previously aligned images & tiles
# trainer.train_dataset.import_stacks_list(work_dir+'/train_stacks_list.csv')
# trainer.valid_dataset.import_stacks_list(work_dir+'/valid_stacks_list.csv')
# trainer.init_dataloaders()
# trainer.deactivate_optimizer_scheduler()

# # reconstruct some image every 20 epochs for debugging purpose
# index = 62
# noisy_ref  =      img_prefix + index2str(index)   + img_fast_suffix
# noisy_imgs =     [img_prefix + index2str(index-2) + img_fast_suffix]
# noisy_imgs.append(img_prefix + index2str(index-1) + img_fast_suffix)
# noisy_imgs.append(img_prefix + index2str(index)   + img_fast_suffix)
# noisy_imgs.append(img_prefix + index2str(index+1) + img_fast_suffix)
# noisy_imgs.append(img_prefix + index2str(index+2) + img_fast_suffix)

# # Actual training
# lr       = 0.01
# lr_decay = 0.85
# trainer.set_lr(lr)
# for i in range(25):
#     for j in range(20):
#         trainer.optimize()
#     lr *= lr_decay
#     trainer.set_lr(lr)

#     # reconstruct the same image every 20 epochs for debugging purpose
#     trainer.reconstruct(noisy_ref, noisy_imgs, 'test_img_'+str(i*20+j)+'_denoised.tif')

# trainer.optimize() # just to get to epoch 500, which is commited to file
# model_filename = trainer.get_current_model()

# # DENOISING
# denoised_prefix = 'denoised_EB_'
# for index in range(50,76):
#     noisy_ref  =      img_prefix + index2str(index)   + img_fast_suffix # note that the ref for alignment has now fast dwell time
#     noisy_imgs =     [img_prefix + index2str(index-2) + img_fast_suffix]
#     noisy_imgs.append(img_prefix + index2str(index-1) + img_fast_suffix)
#     noisy_imgs.append(img_prefix + index2str(index)   + img_fast_suffix)
#     noisy_imgs.append(img_prefix + index2str(index+1) + img_fast_suffix)
#     noisy_imgs.append(img_prefix + index2str(index+2) + img_fast_suffix)
#     trainer.reconstruct(noisy_ref, noisy_imgs, denoised_prefix+index2str(index)+img_fast_suffix)


# #
# # ------ 3. RESUME TRAINING ------
# #

# # TRAINING
# trainer = Trainer(work_dir, cfg.DN_DEPTH, cfg.DN_NN, device, batchsize=cfg.DN_BATCHSZ, dropout=0.3, lr=0.01, naugment=cfg.DN_NAUGMENT)
# # import previous model
# model_filename = work_dir+'/models/model000019.pth'
# trainer.load_model(model_filename)
# # import previously aligned images & tiles
# trainer.train_dataset.import_stacks_list(work_dir+'/train_stacks_list.csv')
# trainer.valid_dataset.import_stacks_list(work_dir+'/valid_stacks_list.csv')
# trainer.init_dataloaders()
# trainer.deactivate_optimizer_scheduler()

# # reconstruct some image every 20 epochs for debugging purpose
# index = 62
# noisy_ref  =      img_prefix + index2str(index)   + img_fast_suffix
# noisy_imgs =     [img_prefix + index2str(index-2) + img_fast_suffix]
# noisy_imgs.append(img_prefix + index2str(index-1) + img_fast_suffix)
# noisy_imgs.append(img_prefix + index2str(index)   + img_fast_suffix)
# noisy_imgs.append(img_prefix + index2str(index+1) + img_fast_suffix)
# noisy_imgs.append(img_prefix + index2str(index+2) + img_fast_suffix)

# # Actual training
# lr       = 0.08 # unfortunately learning rate has to be inputted manually, it should be saved as part of model
# lr_decay = 0.85
# trainer.set_lr(lr)
# for i in range(24):
#     for j in range(20):
#         trainer.optimize()
#     lr *= lr_decay
#     trainer.set_lr(lr)

#     # reconstruct the same image every 20 epochs for debugging purpose
#     trainer.reconstruct(noisy_ref, noisy_imgs, 'test_img_'+str(i*20+j)+'_denoised.tif')

# trainer.optimize() # just to get to epoch 500, which is commited to file
# model_filename = trainer.get_current_model()

# # DENOISING
# denoised_prefix = 'denoised_EB_'
# for index in range(50,76):
#     noisy_ref  =      img_prefix + index2str(index)   + img_fast_suffix # note that the ref for alignment has now fast dwell time
#     noisy_imgs =     [img_prefix + index2str(index-2) + img_fast_suffix]
#     noisy_imgs.append(img_prefix + index2str(index-1) + img_fast_suffix)
#     noisy_imgs.append(img_prefix + index2str(index)   + img_fast_suffix)
#     noisy_imgs.append(img_prefix + index2str(index+1) + img_fast_suffix)
#     noisy_imgs.append(img_prefix + index2str(index+2) + img_fast_suffix)
#     trainer.reconstruct(noisy_ref, noisy_imgs, denoised_prefix+index2str(index)+img_fast_suffix)


# #
# # ------ 4. DENOISE FROM TRAINED MODEL ------
# #

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
