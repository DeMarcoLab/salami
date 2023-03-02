
PIXSZ = 10 # in nm

# Alignment parameters
AL_TILESZ                   = 768
AL_TILE_DENSITY             = 384
AL_BORDER                   = 256  # padding to mitigate border effects
AL_GAU_SIGMA                = 1.0  # pixels
AL_PREFIX                   = 'img_'
AL_PATCH                    = True # True: non-rigid img registration for supervisedlearning
                                   # should be False for supervised (rigid registration only).

# Denoising parameters
DN_DEPTH      = 5
DN_NN         = 5  # even means unsupervised, odd means supervised
DN_TILESZ     = 128
DN_SUBTILESZ  = 64
DN_BORDER     = AL_BORDER
DN_NAUGMENT   = 0
DN_BATCHSZ    = 100

# Supervised training:
# DN_NN is an odd number (1;3;5), it is the total number of slow dwell time images/slices inputted into the network.
# Typically the central slice ("the reference") is of longer dwell time than the adjacent slices,
# which need to be aligned in a non-rigid manner prior to training (AL_PATCH=True).
# The size and overlap of the tiles/patches used for non-rigid registration are controlled by
# AL_TILESZ=N (for a NxN tile) and AL_TILE_DENSITY: tiles are extracted every AL_TILE_DENSITY.
# Consequently, if AL_TILE_DENSITY < AL_TILESZ the tiles overlap, which provides some robustness to the alignement.

# Unsupervised training:
# DN_NN is an even number (2;4), it is the total number of images/slices inputted into the network.
# All images are of the same dwell, such that the images
# can be aligned in a rigid manner prior to training (AL_PATCH=False)
