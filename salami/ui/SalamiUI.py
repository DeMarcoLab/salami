import logging
import os

import napari
import napari.utils.notifications
from fibsem import acquire, alignment
from fibsem import config as fcfg
from fibsem import constants, milling
from fibsem import utils
from fibsem import utils as futils
from fibsem.microscope import FibsemMicroscope
from fibsem.patterning import FibsemMillingStage
from fibsem.structures import (BeamType, FibsemImage, FibsemPattern,
                               ImageSettings, MicroscopeSettings)
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemMillingWidget import FibsemMillingWidget
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget
from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget
from napari.qt.threading import thread_worker
from PyQt5 import QtWidgets, QtCore

import salami.config as cfg
from salami.structures import SalamiImageSettings, SalamiSettings, Experiment
from salami.ui.qt import SalamiUI
from salami.core import run_salami

from fibsem.ui.utils import _get_directory_ui, _get_file_ui, _get_save_file_ui, _get_text_ui

# get path
from salami import config as cfg


class SalamiUI(SalamiUI.Ui_MainWindow, QtWidgets.QMainWindow):
    update_signal = QtCore.pyqtSignal(dict)

    def __init__(self, viewer: napari.Viewer):
        super(SalamiUI, self).__init__()
        self.setupUi(self)
        self.viewer = viewer
        self.viewer.window._qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window._qt_viewer.dockLayerControls.setVisible(False)


        self.exp: Experiment = None

        self.microscope: FibsemMicroscope = None
        self.settings:MicroscopeSettings = None

        self.salami_settings:SalamiSettings = SalamiSettings(
            n_steps=1,
            step_size=1,
            image = [],
            mill= FibsemMillingStage(),
        )

        self.image_widget: FibsemImageSettingsWidget = None
        self.movement_widget: FibsemMovementWidget = None
        self.milling_widget: FibsemMillingWidget = None

        CONFIG_PATH = os.path.join(fcfg.CONFIG_PATH)
        self.system_widget = FibsemSystemSetupWidget(
                microscope=self.microscope,
                settings=self.settings,
                viewer=self.viewer,
                config_path = CONFIG_PATH,
            )
        
        self.setup_connections()
        self.tabWidget.addTab(self.system_widget, "System")

    def setup_connections(self):

        self.system_widget.set_stage_signal.connect(self.set_stage_parameters)
        self.system_widget.connected_signal.connect(self.connect_to_microscope)
        self.system_widget.disconnected_signal.connect(self.disconnect_from_microscope)

        self.pushButton.clicked.connect(self.push_button_clicked)

        self.actionCreate_Experiment.triggered.connect(self.create_experiment)
        self.actionLoad_Experiment.triggered.connect(self.load_experiment)
        self.actionLoad_Protocol.triggered.connect(self.load_protocol)
        self.actionSave_Protocol.triggered.connect(self.save_protocol)

        self.update_signal.connect(self.update_ui_progress)

        self.pushButton_add_imaging.clicked.connect(self._add_imaging_stage)
        self.pushButton_add_imaging.setStyleSheet("background-color: green")
        self.pushButton_remove_imaging.clicked.connect(self._remove_imaging_stage)
        self.pushButton_remove_imaging.setStyleSheet("background-color: red")
        self.pushButton_update_imaging.clicked.connect(self._update_imaging_stage)
        self.pushButton_update_imaging.setStyleSheet("background-color: blue")
    
    def set_stage_parameters(self):
        if self.microscope is None:
            return
        self.settings.system.stage = self.system_widget.settings.system.stage   # TODO: this doesnt actually update the movement widget
        logging.debug(f"Stage parameters set to {self.settings.system.stage}")
        logging.info("Stage parameters set")  

    def update_ui(self):

        _microscope_connected = bool(self.microscope is not None)
        self.tabWidget.setTabVisible(1, _microscope_connected)
        self.tabWidget.setTabVisible(2, _microscope_connected)
        self.tabWidget.setTabVisible(3, _microscope_connected)













        if self.salami_settings is not None:
            logging.info("update salami settings")

            msg = "Salami Settings:\n"
            msg += f"Number of steps: {self.salami_settings.n_steps}, "
            msg += f"Step size: {self.salami_settings.step_size*constants.SI_TO_NANO:.2f}nm\n"
            msg += f"Alignment: {self.salami_settings._align}, Charge Neutralisation: {self.salami_settings._neutralise}\n\n"

            msg += f"{len(self.salami_settings.image)} Imaging Stages:\n"
            ssi: SalamiImageSettings
            for i, ssi in enumerate(self.salami_settings.image):

                msg += f"Stage {i+1:02d}\n"
                msg += f"image: hfw={ssi.image.hfw*constants.SI_TO_MICRO:.2f}um, "
                msg += f"dwell_time:{ssi.image.dwell_time*constants.SI_TO_MICRO:.2f}us, "
                msg += f"resolution: {ssi.image.resolution}\n"
                msg += f"beam: {ssi.beam.working_distance*constants.SI_TO_MILLI:.2f}mm, "
                msg += f"current: {ssi.beam.beam_current*constants.SI_TO_PICO:.2f}pA, "
                msg += f"voltage: {ssi.beam.voltage*constants.SI_TO_KILO:.2f}kV\n"
                msg += f"detector: {ssi.detector.type}, {ssi.detector.mode}\n\n"

            msg += "Milling: \n"
            msg += f"Pattern: {self.salami_settings.mill.pattern.name}, "
            msg += f"Current: {self.salami_settings.mill.milling.milling_current*constants.SI_TO_PICO:.2f}pA, "

            self.label_imaging.setText(msg)


    def _add_imaging_stage(self):

        self.salami_settings.image.append(SalamiImageSettings())

        self._update_combo_box()
    
    def _remove_imaging_stage(self):

        # remove current index
        current_idx = self.comboBox_imaging.currentIndex()
        self.salami_settings.image.pop(current_idx)

        self._update_combo_box()

    def _update_combo_box(self):
        # update combo box
        self.comboBox_imaging.clear()
        self.comboBox_imaging.addItems([f"Stage {i+1:02d}" for i in range(len(self.salami_settings.image))])

        
    def _update_imaging_stage(self):

        # image settings
        image_settings, detector_settings, beam_settings  = self.image_widget.get_settings_from_ui()

        current_idx = self.comboBox_imaging.currentIndex()
        self.salami_settings.image[current_idx] = SalamiImageSettings(
            image=image_settings,
            beam=beam_settings,
            detector=detector_settings,
        )
        logging.info(f"Updated imaging stage {current_idx+1:02d}")
        self.update_ui()

    def update_salami_settings_from_ui(self):

        # general settings
        self.salami_settings.n_steps = int(self.spinBox_n_steps.value())
        self.salami_settings.step_size = (
            float(self.doubleSpinBox_milling_step_size.value()) * constants.NANO_TO_SI
        )
        self.salami_settings.mill=self.milling_widget.get_milling_stages()[0]


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


        PATH = _get_directory_ui(msg="Select a directory to save the experiment", path=cfg.LOG_PATH, parent=self)
        if PATH == "":
            logging.info("No path selected")
            return

        # get name
        NAME, okPressed = _get_text_ui(msg="Enter a name for the experiment", parent=self)

        if NAME == "" or not okPressed:
            logging.info("No name selected")
            return

        self.exp = Experiment(path=PATH, name=NAME)
        self.exp.save()

    def load_experiment(self):
        print("load experiment")

        PATH = _get_file_ui(msg="Select an experiment file", path=cfg.LOG_PATH, parent=self)

        if PATH == "":
            logging.info("No path selected")
            return
        
        self.exp = Experiment.load(fname=PATH)

    def load_protocol(self):
        print("load protocol")

    def save_protocol(self):
        print("save protocol")

    def update_ui_from_protocol(self):

        if self.settings:

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

        self.update_salami_settings_from_ui()

        worker = self.run_salami()
        worker.returned.connect(self.salami_finished)  # type: ignore
        worker.start()

    @thread_worker
    def run_salami(self):

        run_salami(self.microscope, self.settings, self.salami_settings, parent_ui=self)

    def salami_finished(self):
        self.label_ui_status.setText("Finished.")
        self.pushButton.setEnabled(True)
        self.pushButton.setText("Run Salami")
        self.pushButton.setStyleSheet("background-color: green")

    def update_ui_progress(self, info: dict):

        if info["name"] == "Done":
            self.milling_widget.update_ui([info["milling_stage"]])

        if info["name"] == "Image Update":
            self.image_widget.update_viewer(info["image"].data, name=BeamType.ELECTRON.name)

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

            # load default protocol
            self.settings.protocol = futils.load_protocol(cfg.PROTOCOL_PATH)

            # set values from protocol
            self.update_ui_from_protocol()

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
