from align import *
from trainer import Trainer
import config as cfg
import tqdm
import os


# Online usunpervised training, images not provided

# work directory, needs to exists
work_dir = "denoising"
os.makedirs(work_dir, exist_ok=False)
raw_imgs_dir = "../Images/SEM Image 2/"
img_prefix = raw_imgs_dir + "SEM Image 2 - SliceImage - "
img_fast_suffix = ".tif"

# CUDA device
device = "cuda:1"

# utility function to handle filenaming convention, custom per dataset
def index2str(i):
    if i < 1000:
        str_index = "%03d" % i
    else:
        str_index = "%d" % i
    # str_index = '%d' % i
    return str_index


# INIT
aligner = Aligner(work_dir, cfg.PIXSZ, nthreads=8)
trainer = Trainer(
    work_dir,
    cfg.DN_DEPTH,
    cfg.DN_NN,
    device,
    batchsize=cfg.DN_BATCHSZ,
    dropout=0.5,
    lr=0.01,
    naugment=cfg.DN_NAUGMENT,
)


# reconstruct some image every 20 epochs for debugging purpose
index = 2001
noisy_ref = img_prefix + index2str(index) + img_fast_suffix
noisy_imgs = [img_prefix + index2str(index - 1) + img_fast_suffix]
noisy_imgs.append(img_prefix + index2str(index + 1) + img_fast_suffix)


# depth 2, dropout=0.5

# ALIGNMENT & TILES GENERATIONS
index_list = range(1000, 1500, 20)
for index in tqdm.tqdm(index_list):
    noisy_ref = img_prefix + index2str(index) + img_fast_suffix
    noisy_imgs = [img_prefix + index2str(index - 1) + img_fast_suffix]
    noisy_imgs.append(img_prefix + index2str(index + 1) + img_fast_suffix)
    aligned_ref, aligned_imgs = aligner.align_images(
        noisy_ref, noisy_imgs, align_patch=cfg.AL_PATCH
    )
    train_ref, valid_ref, train_inp, valid_inp = aligner.generate_denoising_tiles(
        aligned_ref, aligned_imgs
    )
    trainer.train_dataset.push_stacks(train_ref, train_inp)
    trainer.valid_dataset.push_stacks(valid_ref, valid_inp)
trainer.train_dataset.write_stacks_list(work_dir + "/train_stacks_list.csv")
trainer.valid_dataset.write_stacks_list(work_dir + "/valid_stacks_list.csv")
# trainer.train_dataset.import_stacks_list(work_dir+'/train_stacks_list.csv')
# trainer.valid_dataset.import_stacks_list(work_dir+'/valid_stacks_list.csv')
trainer.init_dataloaders()
trainer.deactivate_optimizer_scheduler()

trainer.model.set_effective_depth(2)  # !!
lr = 0.01
lr_decay = 0.85
trainer.set_lr(lr)
counter = 0
for i in range(5):
    for j in range(20):
        counter += 1
        trainer.optimize()
    lr *= lr_decay
    trainer.set_lr(lr)
    # reconstruct the same image every 20 epochs for debugging purpose
    trainer.reconstruct(
        noisy_ref, noisy_imgs, "test_img_" + str(counter) + "_denoised.tif"
    )

# depth 3, dropout=0.5
index_list = range(1500, 1800, 20)
for index in tqdm.tqdm(index_list):
    noisy_ref = img_prefix + index2str(index) + img_fast_suffix
    noisy_imgs = [img_prefix + index2str(index - 1) + img_fast_suffix]
    noisy_imgs.append(img_prefix + index2str(index + 1) + img_fast_suffix)
    aligned_ref, aligned_imgs = aligner.align_images(
        noisy_ref, noisy_imgs, align_patch=cfg.AL_PATCH
    )
    train_ref, valid_ref, train_inp, valid_inp = aligner.generate_denoising_tiles(
        aligned_ref, aligned_imgs
    )
    trainer.train_dataset.push_stacks(train_ref, train_inp)
    trainer.valid_dataset.push_stacks(valid_ref, valid_inp)

trainer.model.set_effective_depth(3)  # !!
lr = 0.0085
trainer.set_lr(lr)
for i in range(5):
    for j in range(20):
        counter += 1
        trainer.optimize()
    lr *= lr_decay
    trainer.set_lr(lr)
    # reconstruct the same image every 20 epochs for debugging purpose
    trainer.reconstruct(
        noisy_ref, noisy_imgs, "test_img_" + str(counter) + "_denoised.tif"
    )

# depth 4, dropout=0.5
index_list = range(1800, 2100, 20)
for index in tqdm.tqdm(index_list):
    noisy_ref = img_prefix + index2str(index) + img_fast_suffix
    noisy_imgs = [img_prefix + index2str(index - 1) + img_fast_suffix]
    noisy_imgs.append(img_prefix + index2str(index + 1) + img_fast_suffix)
    aligned_ref, aligned_imgs = aligner.align_images(
        noisy_ref, noisy_imgs, align_patch=cfg.AL_PATCH
    )
    train_ref, valid_ref, train_inp, valid_inp = aligner.generate_denoising_tiles(
        aligned_ref, aligned_imgs
    )
    trainer.train_dataset.push_stacks(train_ref, train_inp)
    trainer.valid_dataset.push_stacks(valid_ref, valid_inp)

trainer.model.set_effective_depth(4)  # !!
lr = 0.007225
trainer.set_lr(lr)
for i in range(5):
    for j in range(20):
        counter += 1
        trainer.optimize()
    lr *= lr_decay
    trainer.set_lr(lr)
    # reconstruct the same image every 20 epochs for debugging purpose
    trainer.reconstruct(
        noisy_ref, noisy_imgs, "test_img_" + str(counter) + "_denoised.tif"
    )


# depth 5, dropout=0.3
index_list = range(2100, 2400, 20)
for index in tqdm.tqdm(index_list):
    noisy_ref = img_prefix + index2str(index) + img_slow_suffix
    noisy_imgs = [img_prefix + index2str(index - 1) + img_fast_suffix]
    noisy_imgs.append(img_prefix + index2str(index + 1) + img_fast_suffix)
    aligned_ref, aligned_imgs = aligner.align_images(
        noisy_ref, noisy_imgs, align_patch=cfg.AL_PATCH
    )
    train_ref, valid_ref, train_inp, valid_inp = aligner.generate_denoising_tiles(
        aligned_ref, aligned_imgs
    )
    trainer.train_dataset.push_stacks(train_ref, train_inp)
    trainer.valid_dataset.push_stacks(valid_ref, valid_inp)

trainer.model.set_effective_depth(5)  # !!
trainer.model.set_update_dropout(0.3)  # !!
lr = 0.00614125
trainer.set_lr(lr)
for i in range(15):
    for j in range(20):
        counter += 1
        trainer.optimize()
    lr *= lr_decay
    trainer.set_lr(lr)
    # reconstruct the same image for debugging purpose
    trainer.reconstruct(
        noisy_ref, noisy_imgs, "test_img_" + str(counter) + "_denoised.tif"
    )

# DENOISING
denoised_prefix = "denoised_SEM Image 2 - SliceImage - "
for index in range(1000, 2400, 10):
    noisy_ref = (
        img_prefix + index2str(index) + img_fast_suffix
    )  # note that the ref for alignment has now fast dwell time
    noisy_imgs = [img_prefix + index2str(index - 1) + img_fast_suffix]
    noisy_imgs.append(img_prefix + index2str(index + 1) + img_fast_suffix)
    trainer.reconstruct(
        noisy_ref, noisy_imgs, denoised_prefix + index2str(index) + img_fast_suffix
    )
