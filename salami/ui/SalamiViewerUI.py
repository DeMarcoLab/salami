# image viewer

import glob
from salami.ui.qt import SalamiViewerUI

import logging
import os
import napari

import tifffile as tf
import zarr
from PyQt5 import QtWidgets
import dask.array as da
import dask_image.imread
from salami import config as cfg


class SalamiViewer(SalamiViewerUI.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, viewer: napari.Viewer):
        super(SalamiViewer, self).__init__()
        self.setupUi(self)
        self.viewer = viewer

        self.setup_connections()

    def setup_connections(self):

        self.pushButton_load_data.clicked.connect(self.load_data)
        self.lineEdit_data_path.setText("/home/patrick/github/salami/salami/output")

    def load_data(self):

        self.viewer.layers.clear()
        data_path = self.lineEdit_data_path.text()
        paths = [cfg.RAW_DIR, cfg.DENOISE_DIR, cfg.SEG_DIR]
        names = "Raw", "Denoise", "Seg"

        # TODO: allow user to select which channels to load

        try:
            for path, name in zip(paths, names):

                # get paths
                filter_text = self.lineEdit_filter.text()

                # load data
                data = dask_image.imread.imread(os.path.join(data_path, path, filter_text))
                data.compute_chunk_sizes()

                # update viewer
                if name == "Seg":
                    self.viewer.add_labels(data=data, name=name)
                else:
                    self.viewer.add_image(data=data, name=name)
                    
            self.viewer.grid.enabled = True

        except Exception as e:
            napari.utils.notifications.show_info(f"Exception: {e}")

        

def main():

    viewer = napari.Viewer(ndisplay=2)
    salmai_viewer = SalamiViewer(viewer=viewer)
    viewer.window.add_dock_widget(
        salmai_viewer, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
