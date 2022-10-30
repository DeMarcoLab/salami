

# image viewer

import glob
from salami.ui.qt import SalamiUI

import logging
import os
import napari

import tifffile as tf
import zarr
from PyQt5 import QtWidgets
import dask.array as da

from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget

class SalamiUI(SalamiUI.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(
        self,viewer: napari.Viewer
    ):
        super(SalamiUI, self).__init__()
        self.setupUi(self)
        self.viewer = viewer

        self.setup_connections()

        self.update_ui()

        self.img_settings = FibsemImageSettingsWidget()
        self.gridLayout.addWidget(self.img_settings, 2, 0)
        self.label_image_settings_placeholder.deleteLater()

    def setup_connections(self):

        self.pushButton.clicked.connect(self.push_button_clicked)

    def push_button_clicked(self):
        print("button pushed")

    def update_ui(self):

        import numpy as np

        eb_image = np.random.random(size=(1024, 1536))
        ib_image = np.random.random(size=(1024, 1536))

        self.eb_layer = self.viewer.add_image(eb_image, name="Electron Image")
        self.ib_layer = self.viewer.add_image(ib_image, name="Ion Image")


        self.eb_layer.mouse_double_click_callbacks.append(self._double_click)
        self.ib_layer.mouse_double_click_callbacks.append(self._double_click)
       
        self.eb_layer.mouse_drag_callbacks.append(self._single_click)
        self.ib_layer.mouse_drag_callbacks.append(self._single_click)


        self.viewer.grid.enabled = True

    # double click to move

    # click on ion to move pattern

    # grid view

    def _double_click(self, layer, event):

        coords = layer.world_to_data(event.position)

        print(f"event: {event.type}, layer: {layer}, coords: {coords}, ")

    def _single_click(self, layer, event):

        if event.type == "mouse_press":

            coords = layer.world_to_data(event.position)

            print(f"event: {event.type}, layer: {layer}, coords: {coords}, ")
        else:
            print(f"event: {event.type}")



def main():

    viewer = napari.Viewer(ndisplay=2)
    salami_ui = SalamiUI(viewer=viewer)
    viewer.window.add_dock_widget(
        salami_ui, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()


