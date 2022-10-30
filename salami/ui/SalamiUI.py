import logging
import os

import napari
import napari.utils.notifications
from fibsem.structures import BeamType
from fibsem.ui import utils as ui_utils
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget
from PyQt5 import QtWidgets
from salami.ui.qt import SalamiUI


class SalamiUI(SalamiUI.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(
        self,viewer: napari.Viewer
    ):
        super(SalamiUI, self).__init__()
        self.setupUi(self)
        self.viewer = viewer

        self.setup_connections()

        self.update_ui()

        # reusable components
        self.image_widget = FibsemImageSettingsWidget()
        self.movement_widget = FibsemMovementWidget()
        
        self.gridLayout_imaging_tab.addWidget(self.image_widget, 0, 0)
        self.gridLayout_movement_tab.addWidget(self.movement_widget, 0, 0)


    def setup_connections(self):

        self.pushButton.clicked.connect(self.push_button_clicked)

    def push_button_clicked(self):
        print("button pushed")

    def update_ui(self):

        import numpy as np

        self.eb_image = np.random.random(size=(1024, 1536))
        self.ib_image = np.random.random(size=(1024, 1536))

        self.eb_layer = self.viewer.add_image(self.eb_image, name=BeamType.ELECTRON.name)
        self.ib_layer = self.viewer.add_image(self.ib_image, name=BeamType.ION.name)


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

        # print(f"event: {event.type}, layer: {layer}, coords: {coords}, ")

        # get the relative coords in the beam image
        coords, beam_type = ui_utils.get_beam_coords_from_click(coords, self.eb_image)

        if beam_type is None:
            napari.utils.notifications.show_info(f"Please click inside image to move.")
            return 

        if beam_type is BeamType.ELECTRON:
            adorned_image = self.eb_image
        if beam_type is BeamType.ION:
            adorned_image = self.ib_image

        print(f"beam_type: {beam_type}, coords: {coords}")

    
    # TODO: disable single click and double click at the same time?
    def _single_click(self, layer, event):

        if event.type == "mouse_press":

            coords = layer.world_to_data(event.position)

            # print(f"event: {event.type}, layer: {layer}, coords: {coords}, ")



def main():

    viewer = napari.Viewer(ndisplay=2)
    salami_ui = SalamiUI(viewer=viewer)
    viewer.window.add_dock_widget(
        salami_ui, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()




# move to position
# setup imaging
# setup milling
# run
# tools