import os, sys, math, cmath
import numpy as np
import pyfftw
from   PIL import Image
import matplotlib.pyplot as plt
from   numba import jit, prange

fftw_float_type = 'float32'
fftw_cmplx_type = 'complex64'
float_type      = np.float32

def set_pyfttw_nthreads(n):
    if n <= 0:
        print('Invalid number of threads: ',n)
        sys.exit(-1)
    pyfftw.config.NUM_THREADS = n

class Frame:
    def __init__(self, nx=-1 ,ny=-1, filename=None):
        if filename != None:
            img = Image.open(filename)
            if img.n_frames != 1:
                print('one frame per image')
            nx = img.width
            ny = img.height
            img.close()
        if nx % 2 != 0 or ny % 2 != 0:
            print("Even sized images only")
            sys.exit(-1)
        self.filename = None
        self.nx       = int(nx)
        self.ny       = int(ny)
        self.n        = self.nx * self.ny
        self.D        = pyfftw.zeros_aligned((nx,ny), dtype=fftw_float_type, order='C', n=None)
        self.FD       = pyfftw.zeros_aligned((nx//2+1,ny), dtype=fftw_cmplx_type, order='C', n=None)
        self.fwdt     = pyfftw.FFTW(self.D,  self.FD, axes=(1,0), threads=pyfftw.config.NUM_THREADS, direction='FFTW_FORWARD',  flags=('FFTW_ESTIMATE',))
        self.bwdt     = pyfftw.FFTW(self.FD, self.D,  axes=(1,0), threads=pyfftw.config.NUM_THREADS, direction='FFTW_BACKWARD', flags=('FFTW_ESTIMATE',))
        self.inFourierDomain = False
        if filename != None:
            self.filename = filename
            self.read(filename)

    def reset(self, inFourierDomain=False):
        self.D[:,:] = 0.0
        self.inFourierDomain = inFourierDomain

    def copy(self):
        target = Frame(self.nx, self.ny)
        target.filename = self.filename
        target.D[:,:]   = self.D
        target.FD[:,:]  = self.FD
        target.inFourierDomain = self.inFourierDomain
        return target

    def read(self, filename=None):
        if self.inFourierDomain:
            print("Frame object is being read (real space) while in the Fourier Domain: "+filename)
            sys.exit(-1)
        img = Image.open(filename)
        if img.n_frames != 1:
            print('one frame per image')
            sys.exit(-1)
        if  self.nx != img.width:
            print('Wrong width')
            sys.exit(-1)
        if  self.ny != img.height:
            print('Wrong height')
            sys.exit(-1)                
        self.filename = filename
        img = img.convert('F')
        if self.filename.endswith('.tif') or self.filename.endswith('.tiff'):
            self.D[:,:] = np.rot90(np.array(img, dtype=float_type),-1)
        else:
            self.D[:,:] = np.array(img, dtype=float_type)
        img.close()

    def display(self):
        if not self.inFourierDomain:
            ax = plt.subplot()
            im = ax.imshow(np.rot90(self.D,1),cmap='gray')
        else:
            ax = plt.subplot()
            im = ax.imshow(np.absolute(np.rot90(self.FD,1)),cmap='gray')
        plt.colorbar(im)
        plt.show()

    def write(self, filename, mode='float32'):
        if self.inFourierDomain:
            print("Frame object is being written (real space) while in the Fourier Domain: "+filename)
            sys.exit(-1)

        if mode != None:
            if mode == 'uint8':
                D = np.ndarray((self.nx,self.ny),dtype=np.uint8)
                np.clip(self.D, 0, 255, out=D)
            elif mode == 'float32':
                D = self.D.copy()
            else:
                print('Unsupported writing mode: '+mode)
                sys.exit(-1)
        else:
            D = self.D.copy()

        if filename.endswith('.tif') or filename.endswith('.tiff'):
            img = Image.fromarray(np.rot90(D,1))
        else:
            img = Image.fromarray(D)
        img.save(filename)
        img.close()

    def normalize(self, border=0):
        if self.inFourierDomain:
            print("Frame object is being normalized in real space while in the Fourier Domain")
            sys.exit(-1)
        if border <= 1:
            self.D -= self.D.mean()
            var = np.sum(self.D**2.0) / self.n
        else:
            n = self.n - 2*border*(self.nx+self.ny) + 4*border**2
            if n > 1:
                self.D -= self.D[border:-border,border:-border].mean()
                var = np.sum(self.D[border:-border,border:-border]**2.0) / n
            else:
                return None
        if var > 1.0E-16:
            self.D /= math.sqrt(var)

    def fft(self):
        if not self.inFourierDomain:
            self.D[:,:] = np.fft.fftshift(self.D)
            self.fwdt.execute()
            self.FD[:,:] = np.fft.fftshift(self.FD,axes=(1,)) / self.n
            self.inFourierDomain = True

    def ifft(self):
        if self.inFourierDomain:
            self.FD[:,:] = np.fft.fftshift(self.FD,axes=(1,))
            self.bwdt.execute()
            self.D[:,:] = np.fft.fftshift(self.D)
            self.inFourierDomain = False

    def setDomain(self, inFourierDomain):
        self.inFourierDomain = inFourierDomain

    def calc_mean(self, border=0):
        if self.inFourierDomain:
            print('calc_mean for real space only')
            sys.exit(-1)
        else:
            if border > 0:
                n = self.n - 2*border*(self.nx+self.ny) + 4*border**2
                if n > 1:
                    mean = self.D[border:self.nx-border,border:self.ny-border].mean()
                else:
                    mean = 0.0
            else:
                mean = np.mean(self.D)
        return mean

    def calc_variance(self, mean=None, border=0):
        if self.inFourierDomain:
            var  = 2.0*np.vdot(self.FD[1:-1,:],self.FD[1:-1,:]).real
            var +=     np.vdot(self.FD[0,:],   self.FD[0,:]).real
            var +=     np.vdot(self.FD[-1,:],  self.FD[-1,:]).real
        else:
            nborder = self.n - 2*border*(self.nx+self.ny) + 4*border**2
            if mean == None:
                if nborder < self.n:
                    var = np.sum(self.D[border:self.nx-border,border:self.ny-border]**2) / nborder
                else:
                    var = np.sum(self.D**2) / self.n
            else:
                if nborder < self.n:
                    var = np.sum((self.D[border:self.nx-border,border:self.ny-border]-mean)**2) / nborder
                else:
                    var = np.sum((self.D-mean)**2) / self.n
        return var

    def calc_sdev(self, mean=None):
        var = self.calc_variance(mean=mean)
        if var > 1.0e-16:
            sdev = math.sqrt(var)
        else:
            sdev = 1.0
        return sdev

    # destroys target on output
    def correlate(self, target, normalized=False):
        if self.inFourierDomain:
            if normalized:
                var_target = target.calc_variance()
                var_self   = self.calc_variance()
                sig_target = 1.0
                sig_self   = 1.0
                if var_target > 1.0e-16:
                    sig_target = math.sqrt(var_target)
                if var_self > 1.0e-16:
                    sig_self = math.sqrt(var_self)

            target.FD *= np.conj(self.FD)

            if normalized:
                target.FD /= sig_self * sig_target

            target.ifft()
        else:
            print('correlate for real space not implemented yet')
            sys.exit(-1)

    # returns shift that registers self to target
    def get_best_shift(self, target, normalized=False, peakInterpolation=False, mask=None):
        if (not self.inFourierDomain) or (not target.inFourierDomain):
            print('correlate for real space not implemented yet')
            sys.exit(-1)

        self.correlate(target, normalized=normalized)

        center  = np.array([target.nx/2,target.ny/2]).astype(int)
        b       = np.array([math.floor(target.nx/4),math.floor(target.ny/4)]).astype(int)
        coords  = np.unravel_index( np.argmax(
            target.D[center[0]-b[0]:center[0]+b[0]+1, center[1]-b[1]:center[1]+b[1]+1], axis=None),
            (int(2*b[0]+1), int(2*b[1]+1)) )
        coords += center - b
        cc      = target.D[coords[0],coords[1]]
        offset  = coords - center

        if peakInterpolation:
            offset = np.asarray(offset,dtype=float)
            # along x
            alpha = target.D[coords[0]-1,coords[1]]
            beta  = target.D[coords[0],  coords[1]]
            gamma = target.D[coords[0]+1,coords[1]]
            if alpha < beta and gamma < beta:
                denominator = alpha + gamma - 2.0*beta
                if abs(denominator) > 1.0E-16:
                    offset[0] += 0.5 * (alpha-gamma) / denominator
            else:
                print('error in get_best_shift along x!')
                print(coords,alpha,beta,gamma)
                sys.exit(-1)
            # along y
            alpha = target.D[coords[0],coords[1]-1]
            gamma = target.D[coords[0],coords[1]+1]
            if alpha < beta and gamma<beta:
                denominator = alpha + gamma - 2.0*beta
                if abs(denominator) > 1.0E-16:
                    offset[1] += 0.5 * (alpha-gamma) / denominator
            else:
                print('error in get_best_shift along y!')
                print(coords,alpha,beta,gamma)
                sys.exit(-1)
        return cc, offset

    def crop(self, scale, fftwDims=False):
        if scale > 1.0:
            print('crop is meant for downscaling!')
            sys.exit(-1)
        if not self.inFourierDomain:
            print('crop in real space not implemented yet')
            sys.exit(-1)
        nx = int(round(self.nx * scale))
        ny = int(round(self.ny * scale))
        if fftwDims:
            while nx % 2 == 1 or ny % 2 == 1:
                nx = pyfftw.next_fast_len(nx+1)
                ny = pyfftw.next_fast_len(ny+1)
        else:
            if nx % 2 == 1:
                nx += 1
            if ny % 2 == 1:
                ny += 1
        cropped = Frame(nx,ny)
        c       = (np.array(self.FD.shape) / 2).astype(int)
        nh, nk  = cropped.FD.shape
        n       = int(nk/2)
        cropped.FD[:,:] = self.FD[:nh,c[1]-n:c[1]+n]
        c       = (np.array(cropped.FD.shape) / 2).astype(int)
        cropped.inFourierDomain = True
        return cropped, (nx/self.nx, ny/self.ny)

    def clip(self, left, right, bottom, top):
        if self.inFourierDomain:
            print('clip for real space only')
            sys.exit(-1)
        nx = self.nx - right - left
        ny = self.ny - top - bottom
        if left < 0 or left > self.nx:
            print('Invalid dimension along x; clip: ',left)
            sys.exit(-1)
        if right < 0 or right > self.nx or nx < 0:
            print('Invalid dimension along x; clip: ',right)
            sys.exit(-1)
        if bottom < 0 or bottom > self.ny:
            print('Invalid dimension along y; clip: ',bottom)
            sys.exit(-1)
        if top < 0 or top > self.ny or ny < 0:
            print('Invalid dimension along y; clip: ',top)
            sys.exit(-1)
        clipped        = Frame(nx=nx,ny=ny)
        clipped.D[:,:] = self.D[left:self.nx-right,bottom:self.ny-top]
        return clipped       

    # Tukey window
    def taperEdges(self, alpha=0.9):
        if self.inFourierDomain:
            print('taperEdges for real space only')
            sys.exit(-1)
        if alpha < 0.0001 or alpha > 0.9999:
            print('Invalid alpha value in tarperEdges: '+alpha)
            sys.exit(-1)

        @jit(nopython=True)
        def do(D, alpha, nx, ny):
            halpha   = alpha / 2.0
            halphasq = halpha**2.0
            halfpi   = np.pi / 2.0
            A        = halfpi / (0.5-halpha)
            cx       = nx // 2
            cy       = ny // 2
            for i in range(nx):
                disq = ((i-cx) / nx)**2.0
                for j in range(ny):
                    dsq  = disq + ((j-cy) / ny)**2.0
                    if dsq <= halphasq:
                        continue
                    elif dsq >= 0.25:
                        D[i,j] = 0.0
                    elif dsq > halphasq:
                        D[i,j] *= math.cos(A * (math.sqrt(dsq)-halpha))

        do(self.D, alpha, self.nx, self.ny)

    def shift(self, xoffset, yoffset, fillvalue=0.0):
        if self.inFourierDomain:
            nh, nk = self.FD.shape
            cy     = int(nk/2)
            constx = -2.0*np.pi/self.nx
            consty = -2.0*np.pi/self.ny
            sx     = constx * xoffset
            sy     = consty * yoffset
            for i in range(nh):
                self.FD[i,:] *= cmath.exp(complex(0.0,i*sx))
            for j in range(nk):
                self.FD[:,j] *= cmath.exp(complex(0.0,(j-cy)*sy))
        else:
            if type(xoffset) is int and type(yoffset) is int:
                if xoffset != 0:
                    if xoffset < 0:
                        self.D[:xoffset,:] = self.D[-xoffset:,:]
                        self.D[xoffset:,:] = fillvalue
                    else:
                        self.D[xoffset:,:] = self.D[:-xoffset,:]
                        self.D[:xoffset,:] = fillvalue
                if yoffset != 0:
                    if yoffset < 0:
                        self.D[:,:yoffset] = self.D[:,-yoffset:]
                        self.D[:,yoffset:] = fillvalue
                    else:
                        self.D[:,yoffset:] = self.D[:,:-yoffset]
                        self.D[:,:yoffset] = fillvalue
            else:
                print('Incompatible offsets type')
                sys.exit(-1)

    def highpass(self, ghp):
        if not self.inFourierDomain:
            print('highpass for Fourier domain only')
            sys.exit(-1)
        w        = 7    # falloff window size
        nh, nk   = self.FD.shape
        gw       = w / min(self.nx,self.ny)
        radiush  = int(math.ceil((ghp+gw)*self.nx))
        radiusk  = int(math.ceil((ghp+gw)*self.ny))
        ghpsq    = ghp**2
        ghplimsq = (ghp+gw)**2
        cy       = int(nk/2)
        for j in range(cy-radiusk,cy+radiusk+1):
            k    = j - cy
            gksq = (k/self.ny)**2
            for i in range(radiush+1):
                gsq = gksq + (i/self.nx)**2
                if gsq < ghpsq:
                    self.FD[i,j] = 0.0
                else:
                    if gsq <= ghplimsq:
                        self.FD[i,j] *= 0.5 * (1.0-math.cos(np.pi * (math.sqrt(gsq)-ghp)/gw))

    def lowpass(self, glp):
        if not self.inFourierDomain:
            print('lowpass for Fourier domain only')
            sys.exit(-1)

        @jit(nopython=True)
        def do(FD, nx, ny):
            w        = 5    # falloff window size
            nh, nk   = FD.shape
            gw       = w / min(nx,ny)
            glowlim   = glp - gw
            guplim    = min(glp+gw, 0.5) 
            glowlimsq = glowlim**2
            guplimsq  = guplim**2
            quarterpi = np.pi / 4.0
            cy        = int(nk/2)
            for j in range(nk):
                gksq = ((j - cy)/ny)**2
                for i in range(nh):
                    gsq = gksq + (i/nx)**2
                    if gsq < glowlimsq:
                        continue
                    elif gsq > guplimsq:
                        FD[i,j] = 0.0
                    else:
                        FD[i,j] *= math.cos(quarterpi*(1.0 + (math.sqrt(gsq) - glp)/ gw))

        do(self.FD, self.nx, self.ny)

    def calcPSNR(self, img, maxvalue=255):
        if self.inFourierDomain or img.inFourierDomain:
            print('Images must be in real space! Frame.calcPSNR')
            sys.exit(-1)
        if self.nx != img.nx or self.ny != img.ny:
            print('Images must be of same dimensions! Frame.calcPSNR')
            sys.exit(-1)
        mse  = np.sum((self.D - img.D)**2) / self.n
        mse  = max(mse, 1.0e-20)
        psnr = 10 * math.log10(maxvalue**2/mse) # in dB
        return psnr

    # returns spatial frequencies & FRC
    def calcFRC(self1, self2):
        if not self1.inFourierDomain:
            print('self1 need to be in fourier space in calcFRC')
            sys.exit(-1)
        if not self2.inFourierDomain:
            print('self2 need to be in fourier space in calcFRC')
            sys.exit(-1)
        if self1.nx != self2.nx:
            print('self1 & self2 have different dimensions')
            sys.exit(-1)            
        if self1.ny != self2.ny:
            print('self1 & self2 have different dimensions')
            sys.exit(-1)

        @jit(nopython=True)
        def do(A,B,nx,ny):
            ck = ny // 2
            nh = nx // 2 + 1
            nk = ny // 2 + 1
            n  = min(nh,nk)
            nn = min(nx,ny)

            frc    = np.zeros(n)
            var1   = np.zeros(n)
            var2   = np.zeros(n)

            for h in range(1,nh):
                ghsq   = float(h / nx)**2
                for j in range(0,ny):
                    k     = j - ck
                    if h == 0 and k >= 0:
                        continue
                    gk    = k / ny
                    g     = math.sqrt(ghsq+gk*gk)
                    shell = int(round(g*nn))
                    if shell >= n:
                        continue
                    else:
                        var1[shell] += (A[h,j]*A[h,j].conjugate()).real
                        var2[shell] += (B[h,j]*B[h,j].conjugate()).real
                        frc[shell]  += (A[h,j]*B[h,j].conjugate()).real

            frc[0] = 1.0
            for i in range(1,n):
                frc[i] /= math.sqrt(var1[i]*var2[i])

            return frc

        frc = do(self1.FD, self2.FD, self1.nx, self1.ny)
        n   = len(frc)
        gs  = np.arange(0,n)/(2.*n)
        return gs, frc