# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SalamiUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(384, 676)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 8, 0, 1, 1)
        self.label_title = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setObjectName("label_title")
        self.gridLayout.addWidget(self.label_title, 0, 0, 1, 2)
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setDocumentMode(False)
        self.tabWidget.setTabsClosable(False)
        self.tabWidget.setTabBarAutoHide(False)
        self.tabWidget.setObjectName("tabWidget")
        self.general = QtWidgets.QWidget()
        self.general.setObjectName("general")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.general)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_milling_step_size = QtWidgets.QLabel(self.general)
        self.label_milling_step_size.setObjectName("label_milling_step_size")
        self.gridLayout_2.addWidget(self.label_milling_step_size, 2, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem1, 4, 0, 1, 2)
        self.label_imaging_header = QtWidgets.QLabel(self.general)
        self.label_imaging_header.setObjectName("label_imaging_header")
        self.gridLayout_2.addWidget(self.label_imaging_header, 5, 0, 1, 1)
        self.pushButton_remove_imaging = QtWidgets.QPushButton(self.general)
        self.pushButton_remove_imaging.setObjectName("pushButton_remove_imaging")
        self.gridLayout_2.addWidget(self.pushButton_remove_imaging, 6, 1, 1, 1)
        self.label_n_steps = QtWidgets.QLabel(self.general)
        self.label_n_steps.setObjectName("label_n_steps")
        self.gridLayout_2.addWidget(self.label_n_steps, 1, 0, 1, 1)
        self.pushButton_add_imaging = QtWidgets.QPushButton(self.general)
        self.pushButton_add_imaging.setObjectName("pushButton_add_imaging")
        self.gridLayout_2.addWidget(self.pushButton_add_imaging, 6, 0, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.general)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout_2.addWidget(self.pushButton, 11, 0, 1, 2)
        self.comboBox_imaging = QtWidgets.QComboBox(self.general)
        self.comboBox_imaging.setObjectName("comboBox_imaging")
        self.gridLayout_2.addWidget(self.comboBox_imaging, 5, 1, 1, 1)
        self.spinBox_n_steps = QtWidgets.QSpinBox(self.general)
        self.spinBox_n_steps.setMinimum(1)
        self.spinBox_n_steps.setMaximum(10000)
        self.spinBox_n_steps.setProperty("value", 10)
        self.spinBox_n_steps.setObjectName("spinBox_n_steps")
        self.gridLayout_2.addWidget(self.spinBox_n_steps, 1, 1, 1, 1)
        self.doubleSpinBox_milling_step_size = QtWidgets.QDoubleSpinBox(self.general)
        self.doubleSpinBox_milling_step_size.setMaximum(5000.0)
        self.doubleSpinBox_milling_step_size.setSingleStep(1.0)
        self.doubleSpinBox_milling_step_size.setProperty("value", 100.0)
        self.doubleSpinBox_milling_step_size.setObjectName("doubleSpinBox_milling_step_size")
        self.gridLayout_2.addWidget(self.doubleSpinBox_milling_step_size, 2, 1, 1, 1)
        self.pushButton_update_imaging = QtWidgets.QPushButton(self.general)
        self.pushButton_update_imaging.setObjectName("pushButton_update_imaging")
        self.gridLayout_2.addWidget(self.pushButton_update_imaging, 7, 0, 1, 2)
        self.label_experiment = QtWidgets.QLabel(self.general)
        self.label_experiment.setObjectName("label_experiment")
        self.gridLayout_2.addWidget(self.label_experiment, 0, 0, 1, 2)
        self.label_ui_status = QtWidgets.QLabel(self.general)
        self.label_ui_status.setObjectName("label_ui_status")
        self.gridLayout_2.addWidget(self.label_ui_status, 10, 0, 1, 2)
        self.label_settings = QtWidgets.QLabel(self.general)
        self.label_settings.setObjectName("label_settings")
        self.gridLayout_2.addWidget(self.label_settings, 8, 0, 1, 2)
        self.checkBox_align = QtWidgets.QCheckBox(self.general)
        self.checkBox_align.setObjectName("checkBox_align")
        self.gridLayout_2.addWidget(self.checkBox_align, 3, 0, 1, 1)
        self.checkBox_neutralise = QtWidgets.QCheckBox(self.general)
        self.checkBox_neutralise.setObjectName("checkBox_neutralise")
        self.gridLayout_2.addWidget(self.checkBox_neutralise, 3, 1, 1, 1)
        self.tabWidget.addTab(self.general, "")
        self.gridLayout.addWidget(self.tabWidget, 1, 0, 1, 1)
        self.label_instructions = QtWidgets.QLabel(self.centralwidget)
        self.label_instructions.setText("")
        self.label_instructions.setObjectName("label_instructions")
        self.gridLayout.addWidget(self.label_instructions, 2, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 384, 20))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionCreate_Experiment = QtWidgets.QAction(MainWindow)
        self.actionCreate_Experiment.setObjectName("actionCreate_Experiment")
        self.actionLoad_Experiment = QtWidgets.QAction(MainWindow)
        self.actionLoad_Experiment.setObjectName("actionLoad_Experiment")
        self.actionLoad_Protocol = QtWidgets.QAction(MainWindow)
        self.actionLoad_Protocol.setObjectName("actionLoad_Protocol")
        self.actionSave_Protocol = QtWidgets.QAction(MainWindow)
        self.actionSave_Protocol.setObjectName("actionSave_Protocol")
        self.menuFile.addAction(self.actionCreate_Experiment)
        self.menuFile.addAction(self.actionLoad_Experiment)
        self.menuFile.addAction(self.actionSave_Protocol)
        self.menuFile.addAction(self.actionLoad_Protocol)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_title.setText(_translate("MainWindow", "Salami"))
        self.label_milling_step_size.setText(_translate("MainWindow", "Milling Step Size (nm)"))
        self.label_imaging_header.setText(_translate("MainWindow", "Imaging Stage"))
        self.pushButton_remove_imaging.setText(_translate("MainWindow", "Remove Imaging"))
        self.label_n_steps.setText(_translate("MainWindow", "Number of Steps"))
        self.pushButton_add_imaging.setText(_translate("MainWindow", "Add Imaging"))
        self.pushButton.setText(_translate("MainWindow", "Run Salami"))
        self.pushButton_update_imaging.setText(_translate("MainWindow", "Update Imaging Settings"))
        self.label_experiment.setText(_translate("MainWindow", "TextLabel"))
        self.label_ui_status.setText(_translate("MainWindow", "Status"))
        self.label_settings.setText(_translate("MainWindow", "Image"))
        self.checkBox_align.setText(_translate("MainWindow", "Align Image"))
        self.checkBox_neutralise.setText(_translate("MainWindow", "Charge Neutralisation"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.general), _translate("MainWindow", "General"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionCreate_Experiment.setText(_translate("MainWindow", "Create Experiment"))
        self.actionLoad_Experiment.setText(_translate("MainWindow", "Load Experiment"))
        self.actionLoad_Protocol.setText(_translate("MainWindow", "Load Protocol"))
        self.actionSave_Protocol.setText(_translate("MainWindow", "Save Protocol"))
