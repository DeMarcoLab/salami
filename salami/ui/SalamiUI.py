import logging

import napari
import napari.utils.notifications
from fibsem import acquire, alignment, constants, milling, utils
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemMillingSettings,
    FibsemPattern,
    FibsemPatternSettings,
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

        # set values from protocol
        self.update_ui_from_protocol()

    def setup_connections(self):

        self.pushButton.clicked.connect(self.push_button_clicked)

        self.actionCreate_Experiment.triggered.connect(self.create_experiment)
        self.actionLoad_Experiment.triggered.connect(self.load_experiment)
        self.actionLoad_Protocol.triggered.connect(self.load_protocol)
        self.actionSave_Protocol.triggered.connect(self.save_protocol)

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
        print("run salami pushed")
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
        image_settings: ImageSettings = self.image_widget.get_settings_from_ui()

        # milling settings
        pattern_settings: FibsemPatternSettings = (
            self.milling_widget.get_pattern_settings_from_ui()
        )
        milling_settings: FibsemMillingSettings = (
            self.milling_widget.get_milling_settings_from_ui()
        )

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
            milling.setup_milling(self.microscope, mill_settings=milling_settings)
            milling.draw_pattern(self.microscope, pattern_settings)
            milling.run_milling(
                self.microscope,
                milling_current=milling_settings.milling_current,
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
            if pattern_settings.pattern is FibsemPattern.Line:
                pattern_settings.end_y += milling_step_size
                pattern_settings.start_y += milling_step_size
            if pattern_settings.pattern is FibsemPattern.Rectangle:
                pattern_settings.centre_y += milling_step_size

            # ui update
            yield ("Update", step_no, n_steps, eb_image, pattern_settings)

    def salami_finished(self):
        self.label_ui_status.setText("Finished.")
        self.pushButton.setEnabled(True)
        self.pushButton.setText("Run Salami")
        self.pushButton.setStyleSheet("background-color: gray")

    def update_ui(self, info: tuple):
        stage, step, total_steps, eb_image, pattern_settings = info
        self.label_ui_status.setText(f"{stage} {step+1} of {total_steps}")

        if stage == "Update":
            self.image_widget.update_viewer(eb_image.data, name=BeamType.ELECTRON.name)
            self.milling_widget.update_ui(pattern_settings)


def main():

    viewer = napari.Viewer(ndisplay=2)
    salami_ui = SalamiUI(viewer=viewer)
    viewer.window.add_dock_widget(salami_ui, area="right", add_vertical_stretch=False)
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
