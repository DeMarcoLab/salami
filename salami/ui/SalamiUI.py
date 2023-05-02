import logging

import napari
import napari.utils.notifications
from fibsem import acquire, alignment, constants, milling, utils
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemPattern,
    ImageSettings,
)
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemMillingWidget import FibsemMillingWidget
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget
from napari.qt.threading import thread_worker
from PyQt5 import QtWidgets

import salami.config as cfg
from salami.ui.qt import SalamiUI


class SalamiUI(SalamiUI.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, viewer: napari.Viewer):
        super(SalamiUI, self).__init__()
        self.setupUi(self)
        self.viewer = viewer
        self.viewer.window._qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window._qt_viewer.dockLayerControls.setVisible(False)

        self.setup_connections()

        # connect to microscope
        self.microscope, self.settings = utils.setup_session(protocol_path=cfg.PROTOCOL_PATH)  # type: ignore

        # reusable components
        self.image_widget = FibsemImageSettingsWidget(
            microscope=self.microscope,
            image_settings=self.settings.image,
            viewer=self.viewer,
        )
        self.movement_widget = FibsemMovementWidget(
            microscope=self.microscope,
            settings=self.settings,
            viewer=self.viewer,
            image_widget=self.image_widget,
        )
        self.milling_widget = FibsemMillingWidget(
            microscope=self.microscope,
            settings=self.settings,
            viewer=self.viewer,
            image_widget=self.image_widget,
        )

        self.gridLayout_imaging_tab.addWidget(self.image_widget, 0, 0)
        self.gridLayout_movement_tab.addWidget(self.movement_widget, 0, 0)
        self.gridLayout_milling_tab.addWidget(self.milling_widget, 0, 0)

        # disable ui elements that are not used in salami
        self.disable_ui_elements()

        # set values from protocol
        self.update_ui_from_protocol()

    def setup_connections(self):

        self.pushButton.clicked.connect(self.push_button_clicked)

        self.actionCreate_Experiment.triggered.connect(self.create_experiment)
        self.actionLoad_Experiment.triggered.connect(self.load_experiment)
        self.actionLoad_Protocol.triggered.connect(self.load_protocol)
        self.actionSave_Protocol.triggered.connect(self.save_protocol)

    def disable_ui_elements(self):
        # salami specific setup
        # disable adding/removing milling stages
        self.milling_widget.pushButton_add_milling_stage.setEnabled(False)
        self.milling_widget.pushButton_add_milling_stage.hide()
        self.milling_widget.pushButton_remove_milling_stage.setEnabled(False)
        self.milling_widget.pushButton_remove_milling_stage.hide()

        self.milling_widget.add_milling_stage()

        # disable changing pattern type
        self.milling_widget.comboBox_patterns.setEnabled(False)

    def create_experiment(self):
        print("create experiment")

    def load_experiment(self):
        print("load experiment")

    def load_protocol(self):
        print("load protocol")

    def save_protocol(self):
        print("save protocol")

    def update_ui_from_protocol(self):

        # protocol settings
        self.spinBox_n_steps.setValue(int(self.settings.protocol["num_steps"]))
        self.doubleSpinBox_milling_step_size.setValue(
            float(self.settings.protocol["step_size"]) * constants.SI_TO_NANO
        )

        # image settings

        # movement settings

        # milling settings

    def push_button_clicked(self):
        logging.info("run salami pushed")
        self.pushButton.setEnabled(False)
        self.pushButton.setText("Running...")
        self.pushButton.setStyleSheet("background-color: orange")

        # TODO: disable other microscope interactions

        worker = self.run_salami()
        worker.returned.connect(self.salami_finished)  # type: ignore
        worker.yielded.connect(self.update_ui)  # type: ignore
        worker.start()

    @thread_worker
    def run_salami(self):

        # image settings
        image_settings, _, _  = self.image_widget.get_settings_from_ui()

        stage = self.milling_widget.get_milling_stages()[0]

        # TODO: we need to ensure that the patterns are defined first...

        # general settings
        n_steps = int(self.spinBox_n_steps.value())
        milling_step_size = (
            float(self.doubleSpinBox_milling_step_size.value()) * constants.NANO_TO_SI
        )

        # housekeeping
        image_settings.beam_type = BeamType.ELECTRON
        base_label = image_settings.label

        import time

        eb_image: FibsemImage = None  # type: ignore
        for step_no in range(n_steps):
            logging.info(f"---------- STEP {step_no+1} of {n_steps} ----------")

            yield ("Milling", step_no, n_steps, None, None)
            milling.setup_milling(self.microscope, mill_settings=stage.milling)
            milling.draw_patterns(self.microscope, stage.pattern.patterns)
            milling.run_milling(
                self.microscope,
                milling_current=stage.milling.milling_current,
                asynch=False,
            )
            milling.finish_milling(
                self.microscope, imaging_current=self.settings.system.ion.current
            )
            time.sleep(1)

            if eb_image is not None:
                yield ("Aligning", step_no, n_steps, None, None)
                alignment.beam_shift_alignment(
                    self.microscope, image_settings, eb_image
                )
                time.sleep(1)

            yield ("Acquiring", step_no, n_steps, None, None)
            image_settings.label = f"{base_label}_{step_no:06d}"
            eb_image = acquire.new_image(self.microscope, image_settings)
            time.sleep(1)

            # update pattern
            pattern = stage.pattern.patterns[0]
            if pattern.pattern is FibsemPattern.Line:
                pattern.end_y += milling_step_size
                pattern.start_y += milling_step_size
            if pattern.pattern is FibsemPattern.Rectangle:
                pattern.centre_y += milling_step_size

            # ui update
            yield ("Update", step_no, n_steps, eb_image, stage)

    def salami_finished(self):
        self.label_ui_status.setText("Finished.")
        self.pushButton.setEnabled(True)
        self.pushButton.setText("Run Salami")
        self.pushButton.setStyleSheet("background-color: gray")

    def update_ui(self, info: tuple):
        stage, step, total_steps, eb_image, milling_stage = info
        self.label_ui_status.setText(f"{stage} {step+1} of {total_steps}")

        if stage == "Update":
            self.image_widget.update_viewer(eb_image.data, name=BeamType.ELECTRON.name)
            self.milling_widget.update_ui([milling_stage])


def main():

    viewer = napari.Viewer(ndisplay=2)
    salami_ui = SalamiUI(viewer=viewer)
    viewer.window.add_dock_widget(salami_ui, 
                                  area="right",
                                  name="Salami", 
                                  add_vertical_stretch=True)
    napari.run()


if __name__ == "__main__":
    main()


# Features
# - Multiple Imaging Steps
# - Better setup / validation


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
    - path
    - DetectorSettings
    - ImageSettings
    - MicroscopeState
    - FibsemMillingSettings
    - FibsemPatternSettings
    - protocol: dict

Additional Features:
- Additional Imaging stages

"""
