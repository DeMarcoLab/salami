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


class SalamiViewer(SalamiViewerUI.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, viewer: napari.Viewer):
        super(SalamiViewer, self).__init__()
        self.setupUi(self)
        self.viewer = viewer

        self.setup_connections()

    def setup_connections(self):

        self.pushButton_load_data.clicked.connect(self.load_data)

    def load_data(self):

        try:
            # get paths
            data_path = self.lineEdit_data_path.text()
            filter_text = self.lineEdit_filter.text()
            data_name = self.lineEdit_data_name.text()

            # load data
            data = dask_image.imread.imread(os.path.join(data_path, filter_text))
            data.compute_chunk_sizes()

            # update viewer
            if data_name == "seg":
                self.viewer.add_labels(data=data, name=data_name)
            else:
                self.viewer.add_image(data=data, name=data_name)
                
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
