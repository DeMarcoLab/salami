import logging
import os

import napari
import napari.utils.notifications
from fibsem import config as fcfg
from fibsem import constants
from fibsem import utils as futils
from fibsem.microscope import FibsemMicroscope
from fibsem.patterning import FibsemMillingStage
from fibsem.structures import (BeamType, FibsemImage, FibsemPattern,
                               ImageSettings, MicroscopeSettings)
from fibsem.ui import utils as fui
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemMillingWidget import FibsemMillingWidget
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget
from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget
from napari.qt.threading import thread_worker
from PyQt5 import QtCore, QtGui, QtWidgets

import salami.config as cfg
from salami.core import run_salami
from salami.structures import Experiment, SalamiImageSettings, SalamiSettings
from salami.ui.qt import SalamiUI

INSTRUCTIONS = {
    "START": "Instructions:\nPlease create / load an experiment, and connect to the microscope.",
    "SETUP": "Instructions:\nPlease setup your experiment, imaging and milling settings.",
    "SETUP_IMAGING": "Instructions:\nPlease add/update an imaging stage.",
    "SETUP_MILLING": "Instructions:\nPlease add a milling stage.",
    "RUN": "Instructions:\nPress Run Salami to begin.",
}


class SalamiUI(SalamiUI.Ui_MainWindow, QtWidgets.QMainWindow):
    update_signal = QtCore.pyqtSignal(dict)

    def __init__(self, viewer: napari.Viewer):
        super(SalamiUI, self).__init__()
        self.setupUi(self)
        self.viewer = viewer
        self.viewer.window._qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window._qt_viewer.dockLayerControls.setVisible(False)

        self.exp: Experiment = None
        self.worker = None
        self.microscope: FibsemMicroscope = None
        self.settings: MicroscopeSettings = None

        self.image_widget: FibsemImageSettingsWidget = None
        self.movement_widget: FibsemMovementWidget = None
        self.milling_widget: FibsemMillingWidget = None

        CONFIG_PATH = os.path.join(fcfg.CONFIG_PATH)
        self.system_widget = FibsemSystemSetupWidget(
            microscope=self.microscope,
            settings=self.settings,
            viewer=self.viewer,
            config_path=CONFIG_PATH,
        )
        self.tabWidget.addTab(self.system_widget, "System")
        self.setup_connections()

        # set label as logo, scale to 50x50
        logopath = os.path.join(
            os.path.dirname(cfg.SALAMI_PATH), "docs", "img", "logo.png"
        )
        fui._display_logo(logopath, self.label_logo, [75, 75])

        self.update_ui()

    def setup_connections(self):

        # system widget
        self.system_widget.set_stage_signal.connect(self.set_stage_parameters)
        self.system_widget.connected_signal.connect(self.connect_to_microscope)
        self.system_widget.disconnected_signal.connect(self.disconnect_from_microscope)

        self.pushButton.clicked.connect(self.push_button_clicked)

        # actions
        self.actionCreate_Experiment.triggered.connect(self.create_experiment)
        self.actionLoad_Experiment.triggered.connect(self.load_experiment)
        self.actionLoad_Protocol.triggered.connect(self.load_protocol)
        self.actionSave_Protocol.triggered.connect(self.save_protocol)

        self.update_signal.connect(self.update_ui_progress)

        # image stages
        self.pushButton_add_imaging.clicked.connect(self._add_imaging_stage)
        self.pushButton_add_imaging.setStyleSheet("background-color: green")
        self.pushButton_remove_imaging.clicked.connect(self._remove_imaging_stage)
        self.pushButton_remove_imaging.setStyleSheet("background-color: red")
        self.pushButton_update_imaging.clicked.connect(self._update_imaging_stage)
        self.pushButton_update_imaging.setStyleSheet("background-color: blue")

        # ui updates
        self.spinBox_n_steps.valueChanged.connect(self.update_salami_settings_from_ui)
        self.spinBox_n_steps.setKeyboardTracking(False)
        self.doubleSpinBox_milling_step_size.valueChanged.connect(
            self.update_salami_settings_from_ui
        )
        self.doubleSpinBox_milling_step_size.setKeyboardTracking(False)
        self.checkBox_align.stateChanged.connect(self.update_salami_settings_from_ui)
        self.checkBox_neutralise.stateChanged.connect(
            self.update_salami_settings_from_ui
        )

    def set_stage_parameters(self):
        if self.microscope is None:
            return
        self.settings.system.stage = (
            self.system_widget.settings.system.stage
        )  # TODO: this doesnt actually update the movement widget
        logging.debug(f"Stage parameters set to {self.settings.system.stage}")
        logging.info("Stage parameters set")

    def update_ui(self):

        _microscope_connected = bool(self.microscope is not None)
        self.tabWidget.setTabVisible(2, _microscope_connected)
        self.tabWidget.setTabVisible(3, _microscope_connected)
        self.tabWidget.setTabVisible(4, _microscope_connected)

        _experiment_loaded = bool(self.exp is not None)
        _protocol_loaded = False
        if _experiment_loaded:
            _protocol_loaded = bool(
                self.exp.settings is not None
            )  # TODO: properly validate if all conditions are met

        self.actionLoad_Protocol.setVisible(_experiment_loaded)
        self.actionSave_Protocol.setVisible(_experiment_loaded)
        self.tabWidget.setTabVisible(0, _experiment_loaded and _microscope_connected)

        _valid_imaging = []
        _valid_milling = False
        if _experiment_loaded:
            self.label_experiment.setText(f"Experiment: {self.exp.name}")

            if self.exp.settings is not None:
                logging.info("update salami settings")

                msg = "Salami Settings:\n"
                msg += f"Number of steps: {self.exp.settings.n_steps}, "
                msg += f"Step size: {self.exp.settings.step_size*constants.SI_TO_NANO:.2f}nm\n"
                msg += f"Alignment: {self.exp.settings._align}, Charge Neutralisation: {self.exp.settings._neutralise}\n\n"

                msg += f"Imaging Stages:\n"
                ssi: SalamiImageSettings
                for i, ssi in enumerate(self.exp.settings.image):

                    msg += f"Stage {i+1:02d}\n"
                    _val = (
                        ssi.image is not None
                        and ssi.beam is not None
                        and ssi.detector is not None
                    )
                    if _val:
                        msg += (
                            f"image: hfw={ssi.image.hfw*constants.SI_TO_MICRO:.2f}um, "
                        )
                        msg += f"dwell_time:{ssi.image.dwell_time*constants.SI_TO_MICRO:.2f}us, "
                        msg += f"resolution: {ssi.image.resolution}\n"
                        msg += f"beam: {ssi.beam.working_distance*constants.SI_TO_MILLI:.2f}mm, "
                        msg += f"current: {ssi.beam.beam_current*constants.SI_TO_PICO:.2f}pA, "
                        msg += (
                            f"voltage: {ssi.beam.voltage*constants.SI_TO_KILO:.2f}kV\n"
                        )
                        msg += f"detector: {ssi.detector.type}, {ssi.detector.mode}\n"
                    else:
                        msg += "Not set. Please update the imaging stage\n"
                    _valid_imaging.append(_val)

                msg += "\nMilling: \n"
                _valid_milling = (
                    self.exp.settings.mill.pattern is not None
                    and self.exp.settings.mill.milling is not None
                    and self.exp.settings.mill.pattern.protocol is not None
                )
                if _valid_milling:
                    msg += f"Pattern: {self.exp.settings.mill.pattern.name}, "
                    msg += f"Width: {self.exp.settings.mill.pattern.protocol['width']*constants.SI_TO_MICRO:.2f}um, "
                    msg += f"Depth: {self.exp.settings.mill.pattern.protocol['depth']*constants.SI_TO_MICRO:.2f}um\n"
                    msg += f"Current: {self.exp.settings.mill.milling.milling_current*constants.SI_TO_PICO:.2f}pA, "
                else:
                    msg += "Not set. Please update the pattern.\n"
                self.label_settings.setText(msg)

        else:
            self.label_experiment.setText("Experiment: None")

        _valid_imaging = all(_valid_imaging) and len(_valid_imaging) > 0
        _valid_protocol = _valid_imaging and _valid_milling

        # instructions
        if not _experiment_loaded or not _microscope_connected:
            self.label_instructions.setText(INSTRUCTIONS["START"])
        elif not _valid_imaging:
            self.label_instructions.setText(INSTRUCTIONS["SETUP_IMAGING"])
        elif not _valid_milling:
            self.label_instructions.setText(INSTRUCTIONS["SETUP_MILLING"])
        elif not _protocol_loaded or not _valid_protocol:
            self.label_instructions.setText(INSTRUCTIONS["SETUP"])
        else:
            self.label_instructions.setText(INSTRUCTIONS["RUN"])

        # status labels, color green if valid/connected, red otherwise
        self.label_status_microscope.setText(
            f"Microscope: {'Connected' if _microscope_connected else 'Disconnected'}"
        )
        self.label_status_microscope.setStyleSheet(
            "background-color: green"
            if _microscope_connected
            else "background-color: orange"
        )
        self.label_status_experiment.setText(
            f"Experiment: {'Loaded' if _experiment_loaded else 'Not Loaded'}"
        )
        self.label_status_experiment.setStyleSheet(
            "background-color: green"
            if _experiment_loaded
            else "background-color: orange"
        )
        self.label_status_imaging.setText(
            f"Imaging: {'Valid' if _valid_imaging else 'Invalid'}"
        )
        self.label_status_imaging.setStyleSheet(
            "background-color: green" if _valid_imaging else "background-color: orange"
        )
        self.label_status_milling.setText(
            f"Milling: {'Valid' if _valid_milling else 'Invalid'}"
        )
        self.label_status_milling.setStyleSheet(
            "background-color: green" if _valid_milling else "background-color: orange"
        )

        # buttons
        self.pushButton.setEnabled(_valid_protocol)
        if _valid_protocol:
            self.pushButton.setStyleSheet("background-color: green")
        else:
            self.pushButton.setStyleSheet("background-color: none")

    def _add_imaging_stage(self):

        self.exp.settings.image.append(SalamiImageSettings())

        self._update_combo_box()

    def _remove_imaging_stage(self):

        # remove current index
        current_idx = self.comboBox_imaging.currentIndex()
        self.exp.settings.image.pop(current_idx)

        self._update_combo_box()

    def _update_combo_box(self):
        # update combo box
        self.comboBox_imaging.clear()
        self.comboBox_imaging.addItems(
            [f"Stage {i+1:02d}" for i in range(len(self.exp.settings.image))]
        )

        self.update_ui()

    def _update_imaging_stage(self):

        # image settings
        (
            image_settings,
            detector_settings,
            beam_settings,
        ) = self.image_widget.get_settings_from_ui()

        current_idx = self.comboBox_imaging.currentIndex()
        self.exp.settings.image[current_idx] = SalamiImageSettings(
            image=image_settings,
            beam=beam_settings,
            detector=detector_settings,
        )
        logging.info(f"Updated imaging stage {current_idx+1:02d}")
        self.update_salami_settings_from_ui()
        self.update_ui()

    def update_salami_settings_from_ui(self):

        # general settings
        self.exp.settings.n_steps = int(self.spinBox_n_steps.value())
        self.exp.settings.step_size = (
            float(self.doubleSpinBox_milling_step_size.value()) * constants.NANO_TO_SI
        )
        self.exp.settings._align = self.checkBox_align.isChecked()
        self.exp.settings._neutralise = self.checkBox_neutralise.isChecked()

        # mill stage
        self.exp.settings.mill = self.milling_widget.get_milling_stages()[0]

        if self.sender() is self.pushButton_update_imaging:
            return

        self.update_ui()

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

        PATH = fui._get_directory_ui(
            msg="Select a directory to save the experiment",
            path=cfg.LOG_PATH,
            parent=self,
        )
        if PATH == "":
            logging.info("No path selected")
            return

        # get name

        # get current date
        from datetime import datetime

        now = datetime.now()
        DATE = now.strftime("%Y-%m-%d-%H-%M")
        NAME, okPressed = fui._get_text_ui(
            msg="Enter a name for the experiment", title="Experiment Name", default=f"salami-{DATE}", parent=self
        )

        if NAME == "" or not okPressed:
            logging.info("No name selected")
            return

        self.exp = Experiment(path=PATH, name=NAME)
        self.exp.save()

        napari.utils.notifications.show_info(f"Created experiment {self.exp.name}")
        self.update_ui()

    def load_experiment(self):
        print("load experiment")

        PATH = fui._get_file_ui(
            msg="Select an experiment file", path=cfg.LOG_PATH, parent=self
        )

        if PATH == "":
            logging.info("No path selected")
            return

        self.exp = Experiment.load(fname=PATH)

        napari.utils.notifications.show_info(f"Loaded experiment {self.exp.name}")
        self.update_ui()

    def load_protocol(self):

        if self.exp is None:
            napari.utils.notifications.show_info(f"Please create an experiment first.")
            return

        PATH = fui._get_file_ui(
            msg="Select a protocol file", path=cfg.PROTOCOL_PATH, parent=self
        )

        if PATH == "":
            napari.utils.notifications.show_info(f"No path selected")
            logging.info("No path selected")
            return

        self.exp.settings = SalamiSettings.__from_dict__(futils.load_yaml(path=PATH))
        napari.utils.notifications.show_info(
            f"Loaded Protocol from {os.path.basename(PATH)}"
        )
        self.update_ui()

    def save_protocol(self):

        # convert exp.settings to dict, save to yaml
        PATH = fui._get_save_file_ui(
            msg="Select a protocol file", path=cfg.PROTOCOL_PATH, parent=self
        )

        if PATH == "":
            logging.info("No path selected")
            return

        futils.save_yaml(path=PATH, data=self.exp.settings.__to_dict__())
        napari.utils.notifications.show_info(
            f"Saved Protocol to {os.path.basename(PATH)}"
        )

    def update_ui_from_settings(self):

        if self.exp.settings:

            # protocol settings
            self.spinBox_n_steps.setValue(int(self.exp.settings.n_steps))
            self.doubleSpinBox_milling_step_size.setValue(
                float(self.exp.settings.step_size) * constants.SI_TO_NANO
            )

            # image settings

            # movement settings

            # milling settings

    def push_button_clicked(self):
        logging.info("run salami pushed")
        self.pushButton.setEnabled(False)

        # TODO: disable other microscope interactions

        # self.update_salami_settings_from_ui()
        self.exp.save()

        if self.worker:
            pass
            # TODO: simplify
            # if self.worker.is_running:
            #     self.worker.quit()
            #     logging.info("Worker is already running, pausing...")
            #     self.pushButton.setText("Salami Pausing...")
            #     self.pushButton.setStyleSheet("color: black; background-color: yellow")
            #     return
            # elif self.worker.is_paused:
            #     logging.info("Worker is paused, resuming...")
            #     self.worker.resume()
            #     self.pushButton.setText("Running...")
            #     self.pushButton.setStyleSheet("background-color: orange")
            #     return
        else:
            self.pushButton.setText("Running...")
            self.pushButton.setStyleSheet("background-color: orange")
            self.worker = self.run_salami()
            self.worker.returned.connect(self.salami_finished)  # type: ignore
            self.worker.start()

    @thread_worker
    def run_salami(self):

        yield run_salami(
            self.microscope, self.settings, self.exp.settings, parent_ui=self
        )

    def salami_finished(self):
        self.label_ui_status.setText("Finished.")
        self.pushButton.setEnabled(True)
        self.pushButton.setText("Run Salami")
        self.pushButton.setStyleSheet("background-color: green")
        self.worker = None

    def update_ui_progress(self, info: dict):

        if info["name"] == "Done":
            self.milling_widget.update_ui([info["milling_stage"]])

        if info["name"] == "Image Update":
            self.image_widget.update_viewer(
                info["image"].data, name=BeamType.ELECTRON.name
            )

        msg = f"{info['name']}: ({info['step']+1}/{info['total_steps']})"
        self.label_ui_status.setText(msg)

    # TODO: move this to system wideget??
    def connect_to_microscope(self):
        self.microscope = self.system_widget.microscope
        self.settings = self.system_widget.settings
        self.update_microscope_ui()
        self.update_ui()

    def disconnect_from_microscope(self):

        self.microscope.disconnect()
        self.microscope = None
        self.settings = None
        self.update_microscope_ui()
        self.update_ui()
        self.image_widget = None
        self.movement_widget = None
        self.milling_widget = None

    def update_microscope_ui(self):

        if self.microscope is not None:
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

            # add widgets to tabs
            self.tabWidget.addTab(self.image_widget, "Image")
            self.tabWidget.addTab(self.movement_widget, "Movement")
            self.tabWidget.addTab(self.milling_widget, "Milling")

            # disable ui elements that are not used in salami
            self.disable_ui_elements()

        else:
            if self.image_widget is None:
                return

            # remove tabs
            self.tabWidget.removeTab(4)
            self.tabWidget.removeTab(3)
            self.tabWidget.removeTab(2)

            self.image_widget.clear_viewer()
            self.image_widget.deleteLater()
            self.movement_widget.deleteLater()
            self.milling_widget.deleteLater()


def main():

    viewer = napari.Viewer(ndisplay=2)
    salami_ui = SalamiUI(viewer=viewer)
    viewer.window.add_dock_widget(
        salami_ui, area="right", name="Salami", add_vertical_stretch=True
    )
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
