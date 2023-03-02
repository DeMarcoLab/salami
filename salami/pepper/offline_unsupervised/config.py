
PIXSZ = 10 # in nm

# Alignment parameters
AL_TILESZ                   = 768
AL_TILE_DENSITY             = 384
AL_BORDER                   = 256  # padding to mitigate border effects
AL_GAU_SIGMA                = 1.0  # pixels
AL_PREFIX                   = 'img_'
AL_PATCH                    = False # True: non-rigid img registration for supervisedlearning
                                   # should be False for supervised (rigid registration only).

# Denoising parameters
DN_DEPTH      = 5
DN_NN         = 4  # even means unsupervised, odd means supervised
DN_TILESZ     = 128
DN_SUBTILESZ  = 64
DN_BORDER     = AL_BORDER
DN_NAUGMENT   = 0
DN_BATCHSZ    = 100


