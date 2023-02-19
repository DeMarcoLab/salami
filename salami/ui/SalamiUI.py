import logging
import os

import napari
import napari.utils.notifications
from fibsem.structures import ImageSettings, FibsemPatternSettings, FibsemImage
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget
from fibsem.ui.FibsemMillingWidget import FibsemMillingWidget
from fibsem import constants
from PyQt5 import QtWidgets
from salami.ui.qt import SalamiUI


class SalamiUI(SalamiUI.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(
        self,viewer: napari.Viewer
    ):
        super(SalamiUI, self).__init__()
        self.setupUi(self)
        self.viewer = viewer
        self.viewer.window._qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window._qt_viewer.dockLayerControls.setVisible(False)

        self.setup_connections()
        
        # reusable components
        self.image_widget = FibsemImageSettingsWidget(viewer=self.viewer)
        self.movement_widget = FibsemMovementWidget(viewer=self.viewer, image_widget=self.image_widget)
        self.milling_widget = FibsemMillingWidget(viewer=self.viewer, image_widget=self.image_widget)
        
        self.gridLayout_imaging_tab.addWidget(self.image_widget, 0, 0)
        self.gridLayout_movement_tab.addWidget(self.movement_widget, 0, 0)
        self.gridLayout_milling_tab.addWidget(self.milling_widget, 0, 0)
        

    def setup_connections(self):

        self.pushButton.clicked.connect(self.push_button_clicked)

    def push_button_clicked(self):
        print("button pushed")

        # image settings
        image_settings: ImageSettings = self.image_widget.get_settings_from_ui()
        # pattern settings
        pattern_settings: FibsemPatternSettings = self.milling_widget.get_pattern_settings_from_ui()

        print(image_settings)
        print(pattern_settings)

        # general settings 

        n_steps = int(self.spinBox_n_steps.value())
        milling_step_size = float(self.doubleSpinBox_milling_step_size.value()) * constants.NANO_TO_SI
        
        import time

        for step_no in range(n_steps):
            print(f"------------ \t SLICE {step_no} \t ------------")
            
            time.sleep(1)

            print(f"------------ \t ALIGN {step_no} \t ------------")
            time.sleep(1)

            print(f"------------ \t VIEW  {step_no} \t ------------")
            time.sleep(1)

            # update pattern
            pattern_settings.centre_y += milling_step_size # TODO: make this work for line too

            # TODO: update ui here too.. multi-thread whole program probs


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