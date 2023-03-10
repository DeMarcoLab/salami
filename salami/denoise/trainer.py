import sys, os, time, math, tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from model import SqueezeUNet
from datasets import CustomDataset
from PIL import Image
import config as cfg
import matplotlib.pyplot as plt

# plt.switch_backend('agg')
from align import *
from frame import *

# ###########
DEBUG = False
if DEBUG:
    import mrcfile as mrc
# ###########


class Trainer:
    def __init__(
        self,
        work_dir,
        depth,
        NN,
        device,
        batchsize=50,
        dropout=0.5,
        lr=0.01,
        seed=666,
        naugment=0,
    ):
        if os.path.isdir(work_dir):
            self.work_dir = work_dir
        else:
            print("Work Directory: " + work_dir + " does not exist!")
            sys.exit(-1)
        self.NN = NN
        self.device = device
        self.depth = depth
        self.batchsize = batchsize
        self.dropout = dropout
        self.models_dir = self.work_dir + "/models/"
        self.denoise_dir = self.work_dir + "/denoised/"
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.denoise_dir, exist_ok=True)
        torch.manual_seed(seed)
        self.last_model_written = None

        # model
        self.model = SqueezeUNet(
            NN=self.NN, depth=self.depth, wf=6, dropout=self.dropout
        ).to(self.device)

        # logging
        self.train_epoch_loss = []
        self.val_epoch_loss = []
        self.running_train_loss = []
        self.running_val_loss = []
        self.epoch = 0
        self.freq_write_model = 10

        # Datasets & dataloaders
        self.transform = transforms.ToTensor()
        self.naugment = naugment
        self.train_dataset = CustomDataset(
            NN=cfg.DN_NN, transform=self.transform, augment=True, naugment=self.naugment
        )
        self.valid_dataset = CustomDataset(
            NN=cfg.DN_NN, transform=self.transform, augment=False, naugment=0
        )
        self.train_loader = None
        self.valid_loader = None
        self.dataloader_init = False

        # optimizer & learning rate
        self.lr_min = 0.0005
        self.lr_sched_lambda = 0.99
        self.lr = lr
        self.lr_start = self.lr
        self.lr_sched_active = True
        lambda1 = lambda epoch: self.lr_sched_lambda**epoch
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr
        )
        self.optimizer_scheduler = lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda1
        )

        # loss
        self.loss = nn.MSELoss()

        # Display
        self.figure = None
        self.figs_dir = self.work_dir + "/figures/"
        os.makedirs(self.figs_dir, exist_ok=True)
        self.fig_loss_fname = os.path.join(self.figs_dir, "current_losses.png")
        self.fig_tile_fname = os.path.join(self.figs_dir, "current_tiles.png")

        # summary
        num_parameters = (
            sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1.0e6
        )
        print(f"Model parameters (M): {num_parameters}")
        mem_params = sum(
            [
                param.nelement() * param.element_size()
                for param in self.model.parameters()
            ]
        )
        mem_bufs = sum(
            [buf.nelement() * buf.element_size() for buf in self.model.buffers()]
        )
        mem = mem_params + mem_bufs  # in bytes
        print("Model memory (Gb)  : ", mem / 1024.0**3, flush=True)
        print(f"loss function       : {self.loss}")
        print(f"learning rate       : {self.lr}")

    def init_dataloaders(self):
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batchsize,
            prefetch_factor=6,
            num_workers=12,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batchsize,
            prefetch_factor=6,
            num_workers=12,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
        )
        self.dataloader_init = True
        print("# of training   tiles: ", len(self.train_dataset))
        print("# of validation tiles: ", len(self.valid_dataset))

    def deactivate_optimizer_scheduler(self):
        self.lr_sched_active = False
        print("Optimizer scheduler will not be used")

    def activate_optimizer_scheduler(self):
        self.lr_sched_active = True
        print("Optimizer scheduler will be used")

    def set_lr(self, lr, min_lr=0.0005):
        self.lr = max(lr, min_lr)
        for g in self.optimizer.param_groups:
            g["lr"] = self.lr
        print("Updated learning rate to: %8.5f" % self.lr)

    def __get_optimizer_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def optimize(self):
        print("\n--- EPOCH {} ---".format(self.epoch))
        # learning rate scheduler update
        if self.lr_sched_active:
            lr = self.optimizer_scheduler.get_last_lr()[0]
            if lr < self.lr_min:
                self.lr_sched_active = False
        else:
            lr = self.__get_optimizer_lr()
        # Training
        print("--- Training ---", flush=True)
        print("TRAINING, LR: %8.4f" % lr)
        epoch_train_start_time = time.time()
        self.model.train()
        for batch_id, (targets, inputs) in enumerate(self.train_loader):
            batch_start_time = time.time()
            targets = targets.to(self.device)
            inputs = inputs.to(self.device)
            self.optimizer.zero_grad()
            recs = self.model(inputs)
            loss = self.loss(recs, targets)
            self.running_train_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()

            if (batch_id + 1) % 10 == 0:
                batch_time = time.time() - batch_start_time
                m, s = divmod(batch_time, 60)
                print(
                    "train loss @batch_id {}/{}: {} in {} mins {} secs / batch".format(
                        str(batch_id + 1).zfill(len(str(len(self.train_loader)))),
                        len(self.train_loader),
                        loss.item(),
                        int(m),
                        round(s, 2),
                    ),
                    flush=True,
                )
        self.train_epoch_loss.append(np.array(self.running_train_loss).mean())
        epoch_train_time = time.time() - epoch_train_start_time
        m, s = divmod(epoch_train_time, 60)
        h, m = divmod(m, 60)
        print(
            "epoch training time: {} hrs {} mins {} secs".format(int(h), int(m), int(s))
        )

        if self.lr_sched_active:
            self.optimizer_scheduler.step()

        # Validation
        print("--- Validation ---", flush=True)
        epoch_val_start_time = time.time()
        self.model.eval()
        with torch.no_grad():
            for batch_id, (targets, inputs) in enumerate(self.valid_loader):
                targets = targets.to(self.device)
                inputs = inputs.to(self.device)
                recs = self.model(inputs)
                loss = self.loss(recs, targets)
                self.running_val_loss.append(loss.item())

                if batch_id == 0:
                    # for display
                    rec_batch = recs.detach().cpu().numpy()
                    target_batch = targets.detach().cpu().numpy()
                    input_batch = inputs.detach().cpu().numpy()
                    self.__generate_tiles_image(input_batch, target_batch, rec_batch)

                if (batch_id + 1) % 10 == 0:
                    print(
                        "val loss   @batch_id {}/{}: {}".format(
                            str(batch_id + 1).zfill(len(str(len(self.valid_loader)))),
                            len(self.valid_loader),
                            loss.item(),
                        ),
                        flush=True,
                    )
        self.val_epoch_loss.append(np.array(self.running_val_loss).mean())
        epoch_val_time = time.time() - epoch_val_start_time
        m, s = divmod(epoch_val_time, 60)
        h, m = divmod(m, 60)
        print(
            "epoch validation time: {} hrs {} mins {} secs".format(
                int(h), int(m), int(s)
            ),
            flush=True,
        )

        # Logging
        if self.epoch % self.freq_write_model == 0:
            self.write_model(
                self.models_dir + "model{}.pth".format(str(self.epoch - 1).zfill(6))
            )
        self.__generate_log_image()
        # self.__update_figure()
        self.epoch += 1

    def write_model(self, filename):
        self.model.write_model(filename)
        self.last_model_written = filename
        # torch.save({'model_state_dict': self.model.state_dict(),
        #     'NN'             : self.model.NN,
        #     'nf'             : self.model.nf,
        #     'dropout'        : self.model.dropout,
        #     'depth'          : self.model.depth,
        #     'effective_depth': self.model.effective_depth,
        #     'losses': {'running_train_loss': self.running_train_loss,
        #     'running_val_loss': self.running_val_loss,
        #     'train_epoch_loss': self.train_epoch_loss,
        #     'val_epoch_loss': self.val_epoch_loss},
        #     'nepochs': self.epoch},
        #     self.models_dir + 'model{}.pth'.format(str(self.epoch-1).zfill(6)))

    def load_model(self, filename):
        self.model = self.model.load_model(filename).to(self.device)

    def get_current_model(self):
        return self.last_model_written

    def __update_figure(self):
        if self.figure == None:
            plt.ion()
            self.figure = plt.subplots(2)
            self.figure[1][0].set_axis_off()
            self.figure[1][1].set_axis_off()
            self.figure[0].suptitle("Nothing to show yet")
        losses = np.array(Image.open(self.fig_loss_fname))
        tiles = np.array(Image.open(self.fig_tile_fname))
        self.figure[1][0].cla()
        self.figure[1][1].cla()
        self.figure[1][0].set_axis_off()
        self.figure[1][1].set_axis_off()
        self.figure[1][0].imshow(losses, cmap="gray")
        self.figure[1][1].imshow(tiles, cmap="gray")
        plt.draw()

    def __generate_tiles_image(self, inps, refs, recs):
        b, c, w, h = inps.shape
        n = min(b, 10)
        fig, axs = plt.subplots(n, 3)
        for i in range(n):
            # axs[i,0].imshow(inps[i,:,:,:].sum(axis=0),cmap='gray')
            axs[i, 0].imshow(inps[i, (c - 1) // 2, :, :], cmap="gray")
            axs[i, 1].imshow(refs[i, 0, :, :], cmap="gray")
            axs[i, 2].imshow(recs[i, 0, :, :], cmap="gray")
            axs[i, 0].set_axis_off()
            axs[i, 1].set_axis_off()
            axs[i, 2].set_axis_off()
        plt.savefig(self.fig_tile_fname)
        plt.close(fig)

        # data = np.zeros((b,64,64), dtype=np.float32)
        # data[:,:,:] = inps[:,:,:,:].sum(axis=1)
        # stack = mrc.new("inp0.mrc", overwrite=True)
        # stack.set_data(data)
        # stack.close()

        # data[:,:,:] = inps[:,1,:,:]
        # stack = mrc.new("inp1.mrc", overwrite=True)
        # stack.set_data(data)
        # stack.close()

        # data[:,:,:] = inps[:,2,:,:]
        # stack = mrc.new("inp2.mrc", overwrite=True)
        # stack.set_data(data)
        # stack.close()

        # data[:,:,:] = recs[:,0,:,:]
        # stack = mrc.new("rec.mrc", overwrite=True)
        # stack.set_data(data)
        # stack.close()

        # data[:,:,:] = refs[:,0,:,:]
        # stack = mrc.new("ref.mrc", overwrite=True)
        # stack.set_data(data)
        # stack.close()

    def __generate_log_image(self):
        # losses
        fig = plt.figure(figsize=(16, 16))
        fig.suptitle("Losses", fontsize=20)
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        ax1.title.set_text("epoch train loss VS #epochs")
        ax1.set_xlabel("#epochs")
        ax1.set_ylabel("epoch train loss")
        ax1.plot(self.train_epoch_loss[:])
        ax2.title.set_text("epoch val loss VS #epochs")
        ax2.set_xlabel("#epochs")
        ax2.set_ylabel("epoch val loss")
        ax2.plot(self.val_epoch_loss[:])
        ax3.title.set_text("batch train loss VS #batches")
        ax3.set_xlabel("#batches")
        ax3.set_ylabel("batch train loss")
        ax3.plot(self.running_train_loss[:])
        ax4.title.set_text("batch val loss VS #batches")
        ax4.set_xlabel("#batches")
        ax4.set_ylabel("batch val loss")
        ax4.plot(self.running_val_loss[:])
        plt.savefig(self.fig_loss_fname)
        plt.close(fig)

    def reconstruct(self, noisy_ref, noisy_imgs, filename_rec, path=None):
        if len(noisy_imgs) != self.NN:
            print(
                "The model was not initiated with the correct number of neighbouring slices"
            )
            sys.exit(-1)
        aligner = Aligner(self.denoise_dir, cfg.PIXSZ, nthreads=8)
        # align_patch is always false because now the images are all of identical dwell time
        aligned_ref, aligned_imgs = aligner.align_images(
            noisy_ref, noisy_imgs, align_patch=False
        )

        # SUBTILE_SIZE = cfg.DN_SUBTILESZ
        # BORDER       = 8
        # OFFSET       = 8
        SUBTILE_SIZE = cfg.DN_SUBTILESZ
        BORDER = 0
        OFFSET = 8
        mask = Frame(nx=cfg.DN_SUBTILESZ, ny=cfg.DN_SUBTILESZ)
        mask.D[:, :] = 1.0
        mask.taperEdges(0.90)

        # read in images
        ref_img = np.array(Image.open(aligned_ref), dtype=np.float32)
        nx, ny = ref_img.shape
        mics = np.ndarray((self.NN, nx, ny), dtype=np.float32)
        i = 0
        for filename in aligned_imgs:
            mic = np.array(Image.open(filename), dtype=np.float32)
            mics[i, :, :] = mic
            i += 1

        # tiles coordinates
        ntiles = 0
        tiles_coords = []
        for iy in range(0, ny - SUBTILE_SIZE + 1, OFFSET):
            for ix in range(0, nx - SUBTILE_SIZE + 1, OFFSET):
                ntiles += 1
                tiles_coords.append([ix, iy])

        means = np.ndarray((self.batchsize, self.NN), dtype=np.float32)
        mic_out = np.zeros((nx, ny), dtype=np.float32)
        mic_norm = np.zeros((nx, ny), dtype=np.float32)
        batch = np.zeros(
            (self.batchsize, self.NN, SUBTILE_SIZE, SUBTILE_SIZE), dtype=np.float32
        )
        batch_out = np.zeros(
            (self.batchsize, 1, SUBTILE_SIZE, SUBTILE_SIZE), dtype=np.float32
        )

        self.model.eval()
        for itile in tqdm.tqdm(range(0, ntiles, self.batchsize)):
            batch[:, :, :, :] = 0.0
            ibatch_start = itile
            ibatch_end = min([itile + self.batchsize, ntiles])
            nbatch = ibatch_end - ibatch_start

            for i in range(nbatch):
                ix = tiles_coords[ibatch_start + i][0]
                iy = tiles_coords[ibatch_start + i][1]
                batch[i, :, :, :] = mics[
                    :, ix : ix + SUBTILE_SIZE, iy : iy + SUBTILE_SIZE
                ]

            np.mean(batch[:nbatch, :, :, :], axis=(2, 3), out=means[:nbatch, :])
            batch[:nbatch, :, :, :] -= means[:nbatch, :, None, None]
            noisy_imgs = torch.from_numpy(batch[:nbatch, :, :, :]).to(self.device)

            with torch.no_grad():
                batch_out[:nbatch, :, :, :] = (
                    self.model(noisy_imgs).detach().cpu().numpy()
                )

            batch_out[:nbatch, 0, :, :] += means[:nbatch, 0, None, None]

            for i in range(nbatch):
                ix = tiles_coords[ibatch_start + i][0]
                iy = tiles_coords[ibatch_start + i][1]
                mic_out[
                    ix + BORDER : ix + SUBTILE_SIZE - BORDER,
                    iy + BORDER : iy + SUBTILE_SIZE - BORDER,
                ] += (
                    mask.D
                    * batch_out[
                        i,
                        0,
                        BORDER : SUBTILE_SIZE - BORDER,
                        BORDER : SUBTILE_SIZE - BORDER,
                    ]
                )
                mic_norm[
                    ix + BORDER : ix + SUBTILE_SIZE - BORDER,
                    iy + BORDER : iy + SUBTILE_SIZE - BORDER,
                ] += mask.D

        mic_norm[mic_norm < 0.0001] = 1.0
        mic_out /= mic_norm
        np.clip(mic_out, 0, 255, out=mic_out)
        img_out = Image.fromarray(mic_out.astype("uint8"))
        
        if path is None:
            path = self.denoise_dir
        fname = os.path.join(path, filename_rec)
        print("path: ", fname)
        img_out.save(fname)
