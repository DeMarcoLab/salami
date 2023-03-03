import os, sys
import math
import numpy as np
from frame import *
from numba import jit
from skimage.filters import gaussian
import config as cfg


class Aligner:
    def __init__(self, work_directory, pixsz, nthreads=1):

        # Global parameters
        if os.path.isdir(work_directory):
            self.WORK_DIR = work_directory
        else:
            print("Work Directory: " + work_directory + " does not exist!")
            sys.exit(-1)
        self.PIXSZ = float(pixsz)
        self.LOWPASS_LIMIT = 4.0 * self.PIXSZ
        self.NTHREADS = nthreads
        self.AL_TILESZ = cfg.AL_TILESZ
        self.AL_GLOBAL_COUNTER = -1
        self.AL_TILE_DENSITY = cfg.AL_TILE_DENSITY
        self.AL_BORDER = cfg.AL_BORDER
        self.GAU_SIGMA = cfg.AL_GAU_SIGMA
        self.PREFIX = cfg.AL_PREFIX
        self.DN_GLOBAL_COUNTER = -1
        self.DN_BORDER = cfg.DN_BORDER
        self.DN_TILESZ = cfg.DN_TILESZ
        self.aligned_dir = self.WORK_DIR + "/img_pairs/"
        os.makedirs(self.aligned_dir, exist_ok=True)

        # directories for denoising
        self.data_dir = self.WORK_DIR + "/data/"
        os.makedirs(self.data_dir, exist_ok=True)
        self.train_dir = self.data_dir + "/train/"
        self.valid_dir = self.data_dir + "/valid/"
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.valid_dir, exist_ok=True)

        # TILES
        # parameters
        spatial_freqs = np.fft.rfftfreq(self.AL_TILESZ)
        lowpass_index = np.argmax(spatial_freqs > self.PIXSZ / self.LOWPASS_LIMIT)
        g_lp = lowpass_index / self.AL_TILESZ
        g_hp = 4 / self.AL_TILESZ
        # Convenience objects
        self.tile1 = Frame(nx=self.AL_TILESZ, ny=self.AL_TILESZ)
        self.tile2 = Frame(nx=self.AL_TILESZ, ny=self.AL_TILESZ)
        # resolution soft mask
        lp_mask = Frame(nx=self.AL_TILESZ, ny=self.AL_TILESZ)
        lp_mask.setDomain(True)
        lp_mask.FD[:, :] = 1.0
        lp_mask.lowpass(g_lp)
        lp_mask = np.real(lp_mask.FD)
        hp_mask = Frame(nx=self.AL_TILESZ, ny=self.AL_TILESZ)
        hp_mask.setDomain(True)
        hp_mask.FD[:, :] = 1.0
        hp_mask.highpass(g_hp)
        hp_mask = np.real(hp_mask.FD)
        self.resolution_mask = lp_mask * hp_mask
        # real space soft mask
        self.real_mask = Frame(nx=self.AL_TILESZ, ny=self.AL_TILESZ)
        self.real_mask.D[:, :] = 1.0
        self.real_mask.taperEdges(0.95)

        # the end
        self.INIT = True

    def summary(self):
        print("WORK_DIR      : ", self.WORK_DIR)
        # print('PIXSZ         : ', self.PIXSZ)
        # print('LOWPASS_LIMIT : ', self.LOWPASS_LIMIT)
        # print('TILESZ        : ', self.TILESZ)
        # print('TILE_DENSITY  : ', self.TILE_DENSITY)
        # print('BORDER        : ', self.BORDER)
        # print('GAU_SIGMA     : ', self.GAU_SIGMA)
        # print('GLOBAL_COUNTER: ', self.AL_GLOBAL_COUNTER)
        # print('PREFIX        : ', self.PREFIX)
        print("INIT          : ", self.INIT)
        return None

    def __align_tiles(self, img1, img2, x_range, y_range):
        # individual tiles alignment
        ntx = len(x_range)
        nty = len(y_range)
        x = np.ndarray((ntx, nty))
        y = np.ndarray((ntx, nty))
        sx = np.ndarray((ntx, nty))
        sy = np.ndarray((ntx, nty))
        ni = -1
        for i in x_range:
            ni += 1
            nj = -1
            for j in y_range:
                nj += 1
                self.tile1.setDomain(False)
                self.tile2.setDomain(False)
                self.tile1.D[:, :] = img1.D[
                    i : i + self.AL_TILESZ, j : j + self.AL_TILESZ
                ]
                self.tile2.D[:, :] = img2.D[
                    i : i + self.AL_TILESZ, j : j + self.AL_TILESZ
                ]
                self.tile1.normalize()
                self.tile2.normalize()
                self.tile1.D *= self.real_mask.D
                self.tile2.D *= self.real_mask.D
                self.tile1.fft()
                self.tile2.fft()
                self.tile1.FD *= self.resolution_mask
                self.tile2.FD *= self.resolution_mask
                cc, offset = self.tile1.get_best_shift(
                    self.tile2, normalized=False, peakInterpolation=True
                )
                x[ni, nj], y[ni, nj] = pix2coords(
                    i + self.AL_TILESZ / 2.0, j + self.AL_TILESZ / 2.0, img1.nx, img1.ny
                )
                sx[ni, nj] = offset[0]
                sy[ni, nj] = offset[1]

        # Deformation model fitting
        X = x.flatten()
        Y = y.flatten()
        A = np.array(
            [X * 0.0 + 1.0, X, Y, X * Y, X**2, Y**2, X**2 * Y, X * Y**2]
        ).T
        coeffx, rx, rankx, varx = np.linalg.lstsq(A, sx.flatten(), rcond=None)
        coeffy, ry, ranky, vary = np.linalg.lstsq(A, sy.flatten(), rcond=None)
        return coeffx, coeffy

    def __correct_frame(self, img_in, coeffx, coeffy, img_out):
        nx = img_in.nx
        ny = img_in.ny

        @jit(nopython=True)
        def exec(src, dest):
            for i in range(nx):
                x = (i - (nx / 2)) / nx
                for j in range(ny):
                    y = (j - (ny / 2)) / ny
                    # NN interpolation
                    ci = i - _evalpoly2d(x, y, coeffx)
                    cj = j - _evalpoly2d(x, y, coeffy)
                    fi = min(max(int(round(ci)), 0), nx - 1)
                    fj = min(max(int(round(cj)), 0), ny - 1)
                    dest[i, j] = src[fi, fj]
                    # bilinear interpolation
                    # ci  = i - _evalpoly2d(x,y,coeffx)
                    # fi  = math.floor(ci)
                    # cj  = j - _evalpoly2d(x,y,coeffy)
                    # fj  = math.floor(cj)
                    # if fi < 0 or fi >= nx-2 or fj < 0 or fj >= ny-2:
                    #     # nearest when out of bounds
                    #     fi = min(max(fi,0),nx-1)
                    #     fj = min(max(fj,0),ny-1)
                    #     dest[i,j] = src[fi,fj]
                    # else:
                    #     # bilinear
                    #     di = ci - fi
                    #     dj = cj - fj
                    #     v  = (1.0-di)*( (1.0-dj)*src[fi,fj]   + dj*src[fi,fj+1] )
                    #     v +=      di *( (1.0-dj)*src[fi+1,fj] + dj*src[fi+1,fj+1] )
                    #     dest[i,j] = v

        exec(img_in.D, img_out.D)
        return None

    def align_images(self, ref_filename, imgs_list, align_patch=True):
        if not self.INIT:
            print("Alignment module has not been initialized!")
            sys.exit(-1)
        nimgs = len(imgs_list)
        if ((nimgs - 1) // 2) % 2 < 0:
            print("Invalid number of input images!")
            sys.exit(-1)

        # prep
        self.AL_GLOBAL_COUNTER += 1
        set_pyfttw_nthreads(self.NTHREADS)
        reference = Frame(filename=ref_filename)
        nx = reference.nx
        ny = reference.ny
        corrected_frame = Frame(nx=nx, ny=ny)
        spatial_freqs = np.fft.rfftfreq(reference.nx)
        lowpass_index = np.argmax(spatial_freqs > self.PIXSZ / self.LOWPASS_LIMIT)
        scaling = lowpass_index / len(spatial_freqs)

        # gaussian filtering of reference
        fname_ref_out = "%s/%s%06d_ref.tif" % (
            self.aligned_dir,
            self.PREFIX,
            self.AL_GLOBAL_COUNTER,
        )
        reference_bak = reference.copy()
        reference_bak.write(fname_ref_out, mode="uint8")

        # prep for alignement, interpolation
        reference = reference.clip(
            self.AL_BORDER, self.AL_BORDER, self.AL_BORDER, self.AL_BORDER
        )
        reference.normalize()
        reference.taperEdges(0.95)
        reference.fft()
        reference_crop_bak, scaling_eff = reference.crop(scaling, fftwDims=True)
        reference_crop_bak.highpass(0.003)
        reference_crop_bak.lowpass(0.49)

        aligned_images_list = []
        N = -(nimgs - 1) // 2 - 1
        for img_fname in imgs_list:
            N += 1
            output_name = "%s/%s%06d_%1d.tif" % (
                self.aligned_dir,
                self.PREFIX,
                self.AL_GLOBAL_COUNTER,
                N,
            )

            # read & image prep
            frame = Frame(filename=img_fname)
            frame_bak = frame.copy()
            frame = frame.clip(
                self.AL_BORDER, self.AL_BORDER, self.AL_BORDER, self.AL_BORDER
            )
            frame.normalize()
            frame.taperEdges(0.95)
            frame.fft()
            frame_crop, scaling_eff = frame.crop(scaling, fftwDims=True)
            frame_crop.highpass(0.003)
            frame_crop.lowpass(0.49)

            # Alignment
            reference_crop = reference_crop_bak.copy()
            cc, offset = frame_crop.get_best_shift(
                reference_crop, normalized=False, peakInterpolation=True
            )
            offset /= scaling_eff

            if align_patch:
                # best option when image and reference are of different dwell time
                # Shift image
                frame_bak.shift(int(round(offset[0])), int(round(offset[1])))

                # Tiles partition
                border_x = int(max(self.AL_BORDER, math.ceil(abs(offset[0]))))
                border_y = int(max(self.AL_BORDER, math.ceil(abs(offset[1]))))
                x_range = range(
                    border_x,
                    int(frame_bak.nx - border_x - self.AL_TILESZ),
                    self.AL_TILE_DENSITY,
                )
                y_range = range(
                    border_y,
                    int(frame_bak.ny - border_y - self.AL_TILESZ),
                    self.AL_TILE_DENSITY,
                )

                # Local Alignment
                coeffs_x, coeffs_y = self.__align_tiles(
                    frame_bak, reference_bak, x_range, y_range
                )
                set_pyfttw_nthreads(self.NTHREADS)

                # Interpolation
                self.__correct_frame(frame_bak, coeffs_x, coeffs_y, corrected_frame)

                # write
                corrected_frame.write(output_name, mode="uint8")
            else:
                # sufficient when reference and image are of identical dwell time
                # Shift image
                frame_bak.fft()
                frame_bak.shift(offset[0], offset[1])
                frame_bak.ifft()

                # write
                frame_bak.write(output_name, mode="uint8")

            aligned_images_list.append(output_name)

        return fname_ref_out, aligned_images_list

    def generate_denoising_tiles(
        self, reference, noisy_inputs, blur_reference=True, valid_frac=0.2
    ):
        self.DN_GLOBAL_COUNTER += 1
        ref = Frame(filename=reference)
        if blur_reference:
            A = gaussian(ref.D, sigma=self.GAU_SIGMA, preserve_range=True)
            ref.D[:, :] = A
        ref = ref.clip(self.DN_BORDER, self.DN_BORDER, self.DN_BORDER, self.DN_BORDER)
        ntx = ref.nx // self.DN_TILESZ
        nty = ref.ny // self.DN_TILESZ
        nt = ntx * nty

        ref_tiles = np.ndarray((nt, self.DN_TILESZ, self.DN_TILESZ), dtype=np.float32)
        vec = np.zeros(nt)
        nval = int(round(valid_frac * nt))
        vec[:nval] = 1.0
        np.random.shuffle(vec)

        # Extract reference tiles
        n = -1
        for i in range(ntx):
            pi = i * self.DN_TILESZ
            for j in range(nty):
                pj = j * self.DN_TILESZ
                n += 1
                ref_tiles[n, :, :] = np.rot90(
                    ref.D[pi : pi + self.DN_TILESZ, pj : pj + self.DN_TILESZ], 1
                )

        ref_valid_tiles = []
        ref_train_tiles = []
        for i in range(nt):
            if vec[i] < 0.5:
                if blur_reference:
                    ref_train_tiles.append(Image.fromarray(ref_tiles[i, :, :]))
                else:
                    ref_train_tiles.append(
                        Image.fromarray(ref_tiles[i, :, :].astype("uint8"))
                    )
            else:
                if blur_reference:
                    ref_valid_tiles.append(Image.fromarray(ref_tiles[i, :, :]))
                else:
                    ref_valid_tiles.append(
                        Image.fromarray(ref_tiles[i, :, :].astype("uint8"))
                    )

        # Write reference stack
        train_ref_name = (
            self.train_dir + "train_" + str(self.DN_GLOBAL_COUNTER) + "_ref.tif"
        )
        valid_ref_name = (
            self.valid_dir + "valid_" + str(self.DN_GLOBAL_COUNTER) + "_ref.tif"
        )
        ref_train_tiles[0].save(
            train_ref_name,
            compression=None,
            save_all=True,
            append_images=ref_train_tiles[1:],
        )
        ref_valid_tiles[0].save(
            valid_ref_name,
            compression=None,
            save_all=True,
            append_images=ref_valid_tiles[1:],
        )

        # Extract validation tiles
        inp_train_names = []
        inp_valid_names = []
        half_window = len(noisy_inputs) // 2
        frame_index = -half_window - 1
        for noisy_input in noisy_inputs:
            frame_index += 1
            frame = Frame(filename=noisy_input)
            frame = frame.clip(
                self.DN_BORDER, self.DN_BORDER, self.DN_BORDER, self.DN_BORDER
            )
            inp_tiles = np.ndarray(
                (nt, self.DN_TILESZ, self.DN_TILESZ), dtype=np.float32
            )

            n = -1
            for i in range(ntx):
                pi = i * self.DN_TILESZ
                for j in range(nty):
                    pj = j * self.DN_TILESZ
                    n += 1
                    inp_tiles[n, :, :] = np.rot90(
                        frame.D[pi : pi + self.DN_TILESZ, pj : pj + self.DN_TILESZ], 1
                    )
            inp_valid_tiles = []
            inp_train_tiles = []
            for i in range(nt):
                if vec[i] < 0.5:
                    inp_train_tiles.append(
                        Image.fromarray(inp_tiles[i, :, :].astype("uint8"))
                    )
                else:
                    inp_valid_tiles.append(
                        Image.fromarray(inp_tiles[i, :, :].astype("uint8"))
                    )
            # Write input stack
            train_name = (
                self.train_dir
                + "train_"
                + str(self.DN_GLOBAL_COUNTER)
                + "_"
                + str(frame_index)
                + ".tif"
            )
            valid_name = (
                self.valid_dir
                + "valid_"
                + str(self.DN_GLOBAL_COUNTER)
                + "_"
                + str(frame_index)
                + ".tif"
            )
            inp_train_tiles[0].save(
                train_name,
                compression=None,
                save_all=True,
                append_images=inp_train_tiles[1:],
            )
            inp_valid_tiles[0].save(
                valid_name,
                compression=None,
                save_all=True,
                append_images=inp_valid_tiles[1:],
            )
            inp_train_names.append(train_name)
            inp_valid_names.append(valid_name)

        return train_ref_name, valid_ref_name, inp_train_names, inp_valid_names


# UTILITIES

# to evaluate deformation model
@jit(nopython=True)
def _evalpoly2d(X, Y, c):
    return (
        c[0]
        + X * c[1]
        + Y * c[2]
        + X * Y * c[3]
        + X**2 * c[4]
        + Y**2 * c[5]
        + X**2 * Y * c[6]
        + X * Y**2 * c[7]
    )


# to perform coordinate transform
@jit(nopython=True)
def pix2coords(i, j, nx, ny):
    x = (i - (nx / 2)) / nx
    y = (j - (ny / 2)) / ny
    return x, y
