from frame import *
import matplotlib.pyplot as plt

# how to calculate FRC between to images & calculate PSNR

# read
img1 = Frame(filename='xxx')
img2 = Frame(filename='yyy')

# remove borders
border=256
img1 = img1.clip(border,border,border,border)
img2 = img2.clip(border,border,border,border)

# PSNR
psnr = img1.calcPSNR(img2)

# FRC
# image normalization, masking, Fourier transformation
img1.normalize()
img1.taperEdges(0.95) # rectangular Tukey windowing (= cosine edge)
img1.fft()

img2.normalize()
img2.taperEdges(0.95) # rectangular Tukey windowing (= cosine edge)
img2.fft()

# actual calculation & plot
g, frc = img1.calcFRC(img2)
fig, axs = plt.subplots(1)
axs.plot(g,frc)
plt.show()
