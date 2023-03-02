import numpy as np
import os, sys, math
from torch.utils.data import Dataset
from PIL import Image
import config as cfg
from torchvision import transforms
import matplotlib.pyplot as plt
import csv
import tqdm

class CustomDataset(Dataset):

    def __init__(self, NN=1, transform=None, augment=False, naugment=0):
        self.NN              = NN
        self.transform       = transform
        self.ref_stacks      = []
        self.img_stacks      = []
        self.tiles_per_stack = []
        self.augment         = augment
        self.naugment        = naugment
        self.nstacks         = 0
        self.nimgs           = 0
        self.ncumul          = None

    def push_stacks(self, ref_stack, imgs_stack):
        if len(imgs_stack) != self.NN:
            print('Inconsistent number of stacks!')
            sys.exit(-1)
        img1    = Image.open(ref_stack)
        w, h    = img1.size
        nframes = img1.n_frames
        if w != cfg.DN_TILESZ:
            print('Incorrect width in stack:', ref_stack)
            sys.exit(-1)
        if h != cfg.DN_TILESZ:
            print('Incorrect height in stack:', ref_stack)
            sys.exit(-1)
        for fname in imgs_stack:
            img = Image.open(fname)
            if img.n_frames != nframes:
                print('Inconsistent number of frames in stacks:', fname)
                sys.exit(-1)
            if img.size[0] != w:
                print('Inconsistent width in stacks:', fname)
                sys.exit(-1)
            if img.size[0] != h:
                print('Inconsistent height in stacks:', fname)
                sys.exit(-1)
        self.ref_stacks.append(ref_stack)
        self.img_stacks.append(imgs_stack)
        self.tiles_per_stack.append(nframes)
        self.nstacks += 1
        self.nimgs   += nframes
        self.ncumul   = np.zeros(self.nstacks,dtype=int)
        for i in range(0,self.nstacks):
            self.ncumul[i] = np.sum(self.tiles_per_stack[:i+1])

    def write_stacks_list( self, filename ):
        with open(filename, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            for i in range(self.nstacks):
                row = [self.ref_stacks[i]]
                row.extend(self.img_stacks[i])
                writer.writerow(row)

    def import_stacks_list( self, filename ):
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            n = 0
            for row in tqdm.tqdm(reader):
                self.push_stacks(row[0], row[1:])
                n += 1
        print("Imported %6d stacks" % n)

    # reference, multichannel input
    def __getitem__(self, absolute_img_index):
        img_index   = absolute_img_index//(self.naugment+1)
        stack_index = self.nstacks-1
        for i in range(0,self.nstacks-1):
            if img_index < self.ncumul[i]:
                stack_index = i
                break
        if stack_index == 0:
            index = img_index
        else:
            index = img_index - self.ncumul[stack_index-1]

        # Reference
        img = Image.open(self.ref_stacks[stack_index])
        img.seek(index)
        target = np.array(img.getdata(),dtype=np.float32).reshape(cfg.DN_TILESZ,cfg.DN_TILESZ)
        img.close()

        # translation
        if self.augment:
            x = np.random.randint(0,cfg.DN_TILESZ-cfg.DN_SUBTILESZ)
            y = np.random.randint(0,cfg.DN_TILESZ-cfg.DN_SUBTILESZ)
        else:
            x = y = (cfg.DN_TILESZ-cfg.DN_SUBTILESZ) // 2
        target = target[x:x+cfg.DN_SUBTILESZ, y:y+cfg.DN_SUBTILESZ]

        # rotation, p=0.5
        # rot            = np.random.randint(0,2)
        rot            = 1
        rotation_times = np.random.randint(1,4)
        if self.augment:
            if rot == 1:
                target = np.rot90(target,rotation_times)

        # flip, p=0.5
        # iflip = np.random.randint(0,2)
        iflip = 1
        flip  = np.random.randint(0,2)
        if self.augment:
            if iflip == 1:
                target = np.flip(target,axis=flip)

        # brightness & histogram clipping
        mean    = np.mean(target)
        target -= mean
        np.clip(target, -mean, -mean+255,out=target)

        # Input images
        inps = np.ndarray((cfg.DN_SUBTILESZ,cfg.DN_SUBTILESZ,self.NN),dtype=np.float32)
        for i in range(self.NN):
            img = Image.open(self.img_stacks[stack_index][i])
            img.seek(index)
            inp = np.array(img.getdata(),dtype=np.float32).reshape(cfg.DN_TILESZ,cfg.DN_TILESZ)
            img.close()
            # translation
            inp    = inp[x:x+cfg.DN_SUBTILESZ, y:y+cfg.DN_SUBTILESZ]
            # brightness
            mean    = np.mean(inp)
            inp    -= mean
            # contrast
            if self.augment:
                inp *= 1. + np.random.randint(-5,6) / 5.
            # rotation
            if self.augment and rot == 1:
                inp[:,:] = np.rot90(inp,rotation_times)
            # flip
            if self.augment and iflip == 1:
                inp[:,:] = np.flip(inp, axis=flip)
            # gaussian noise
            if self.augment and self.NN % 2 == 1:
            # if self.augment and np.random.randint(0,2) == 1:
                inp += np.random.normal(0.0, 1.0, (cfg.DN_SUBTILESZ,cfg.DN_SUBTILESZ))
            # histogram clipping
            np.clip(inp, -mean, -mean+255,out=inps[:,:,i])

        # Swap slices
        if self.augment and self.NN > 2 and self.NN % 2 == 1:
            if np.random.randint(0,2) == 1:
                inps_copy = inps.copy()
                for i in range(self.NN):
                    if i != self.NN//2:
                        inps[:,:,i] = inps_copy[:,:,self.NN-1-i]

        return self.transform(target.copy()), self.transform(inps.copy())

    def __len__(self):
        return int((self.naugment+1) * self.nimgs)
