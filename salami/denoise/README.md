# denoise

Deep Learning based denoising for FIBSEM images

offline_protocol.py contains several example drivers to perform all or part of the pipeline using images from the raw_imgs directory

# Background

Volume Electron Microscopy provides a unique opportunity to study the structure of biological samples at the cellular level. However, the data produced by these instruments is often noisy and difficult to interpret. This project aims to develop a deep learning based denoising pipeline to improve the quality of FIBSEM images.

Three core issues:

## Determining optimal collection settings

- Often the optimal collection settings are not known
- Users use too high settings (e.g. dwell time) to ensure good quality images
- However, often no improvement in quality is seen, after a certain level
- Causes unnecessary time and cost, and can damage the sample

## Image Quality vs Time

- Users often have to choose between image quality and time
- However, it is often possible to improve both
- This project aims to develop a deep learning based denoising pipeline to improve the quality of FIBSEM images
- This will allow users to collect images at lower settings, reducing time and cost, while still maintaining good image quality

## Real time feedback on state of collection

- Users often have to wait until the end of a collection to see the results
- This can be frustrating, and can lead to users collecting more images than necessary
- This project aims to develop a deep learning based segmentation pipeline to improve the quality of FIBSEM images
- This will allow users to see the results of their collection in real time, allowing them to to finish collection once they are happy with the results / found the structure they are looking for

# Project

The project is split into two parts:

## Analysis Pipeline

1. Run a parameter sweep to find the best parameters for data collection
2. Fit a curve to produce the best resolution, noise, time tradeoff
3. Collect data using optimal parameters

## Denoising and Segmentation Pipeline

time budget: 12.5 seconds (collection time per image)

1. Read in raw images
2. Preprocess images for denosing (tile)
3. Denoise images with model
4. Post process images (restack, align)
5. Preprocess images for segmentation (?)
6. Segment images with model
7. Post process images (restack, align)
8. Save images

- Support for online training
- Support for offline training
- Support for online visualisation


## Visualisation
Napari