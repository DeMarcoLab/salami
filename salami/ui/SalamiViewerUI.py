

# image viewer

import glob
from salami.ui.qt import SalamiViewerUI

import logging
import os
import napari

import tifffile as tf
import zarr
from PyQt5 import QtWidgets

class SalamiViewer(SalamiViewerUI.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(
        self,viewer: napari.Viewer
    ):
        super(SalamiViewer, self).__init__()
        self.setupUi(self)
        self.viewer = viewer

        self.setup_connections()

    def setup_connections(self):

        self.pushButton_load_data.clicked.connect(self.load_data)


    def load_data(self):

        import dask.array as da

        try:        
            # get paths
            data_path = self.lineEdit_data_path.text()
            filter_text = self.lineEdit_filter.text()

            # load data
            # TODO: fix the sorting
            data = da.from_zarr(tf.imread(os.path.join(data_path, filter_text), aszarr=True, sort=True))

            # update viewer
            # self.viewer.layers.clear()
            self.viewer.add_image(data=data, name=filter_text)
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


