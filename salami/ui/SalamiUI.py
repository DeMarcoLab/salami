import logging
import os

import napari
import napari.utils.notifications
from fibsem import acquire, alignment, constants, milling, utils
from fibsem.structures import (BeamType, FibsemImage, FibsemMillingSettings,
                               FibsemPatternSettings, ImageSettings)
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemMillingWidget import FibsemMillingWidget
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget
from PyQt5 import QtWidgets

import salami.config as cfg
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

        # connect to microscope
        self.microscope, self.settings = utils.setup_session(protocol_path=cfg.PROTOCOL_PATH)
        
        # reusable components
        self.image_widget = FibsemImageSettingsWidget(microscope=self.microscope, viewer=self.viewer)
        self.movement_widget = FibsemMovementWidget(microscope =self.microscope, viewer=self.viewer, image_widget=self.image_widget)
        self.milling_widget = FibsemMillingWidget(microscope=self.microscope, viewer=self.viewer, image_widget=self.image_widget)
        
        self.gridLayout_imaging_tab.addWidget(self.image_widget, 0, 0)
        self.gridLayout_movement_tab.addWidget(self.movement_widget, 0, 0)
        self.gridLayout_milling_tab.addWidget(self.milling_widget, 0, 0)
        

    def setup_connections(self):

        self.pushButton.clicked.connect(self.push_button_clicked)

    def push_button_clicked(self):
        print("button pushed")

        # image settings
        image_settings: ImageSettings = self.image_widget.get_settings_from_ui()

        # milling settings
        pattern_settings: FibsemPatternSettings = self.milling_widget.get_pattern_settings_from_ui()
        milling_settings: FibsemMillingSettings = self.milling_widget.get_milling_settings_from_ui()

        print(image_settings)
        print(pattern_settings)
        print(milling_settings)

        # general settings 
        n_steps = int(self.spinBox_n_steps.value())
        milling_step_size = float(self.doubleSpinBox_milling_step_size.value()) * constants.NANO_TO_SI
        
        # housekeeping
        image_settings.beam_type = BeamType.ELECTRON
        base_label = image_settings.label

        import time
        eb_image: FibsemImage = None
        for step_no in range(n_steps):
            logging.info(f"---------- STEP {step_no+1} of {n_steps} ----------")

            self.update_ui("slice", step_no, n_steps)
            milling.setup_milling(self.microscope, mill_settings=milling_settings)
            milling.draw_pattern(self.microscope, pattern_settings)
            milling.run_milling(self.microscope, milling_current=milling_settings.milling_current, asynch=False)
            milling.finish_milling(self.microscope, imaging_current=self.settings.system.ion.current)
            time.sleep(1)

            if eb_image is not None:
                self.update_ui("align", step_no, n_steps)
                alignment.beam_shift_alignment(self.microscope, image_settings, eb_image)
                time.sleep(1)

            self.update_ui("view", step_no, n_steps)
            image_settings.label = f"{base_label}_{step_no:04d}"
            eb_image = acquire.new_image(self.microscope, image_settings)
            time.sleep(1)

            # update pattern
            pattern_settings.centre_y += milling_step_size # TODO: make this work for line too

            # ui update
            # self.image_widget.update_viewer(eb_image.data, name=eb_image.metadata.image_settings.beam_type.name)
            
            # TODO: update ui here too.. multi-thread whole program probs

        self.label_ui_status.setText("Finished.")

    def update_ui(self, stage, step, total_steps):
        
        self.label_ui_status.setText(f"{stage} {step+1} of {total_steps}")
        self.repaint()



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

"""
TODO:
- Create experiment

- Load Experiment
- Save Protocol
- Load Protocol 

Experiment: 
    - DetectorSettings
    - ImageSettings
    - MicroscopeState
    - FibsemMillingSettings
    - FibsemPatternSettings
    - protocol: dict
"""