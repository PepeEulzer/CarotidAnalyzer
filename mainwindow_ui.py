# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Git\carotidanalyzer\ui\mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1920, 1080)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/resources/centerline.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.empty_central_widget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.empty_central_widget.sizePolicy().hasHeightForWidth())
        self.empty_central_widget.setSizePolicy(sizePolicy)
        self.empty_central_widget.setObjectName("empty_central_widget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.empty_central_widget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(self.empty_central_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        MainWindow.setCentralWidget(self.empty_central_widget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1920, 31))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolbar_modules = QtWidgets.QToolBar(MainWindow)
        self.toolbar_modules.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.toolbar_modules.setMovable(False)
        self.toolbar_modules.setFloatable(False)
        self.toolbar_modules.setObjectName("toolbar_modules")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar_modules)
        self.toolbar_save = QtWidgets.QToolBar(MainWindow)
        self.toolbar_save.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.toolbar_save.setMovable(False)
        self.toolbar_save.setFloatable(False)
        self.toolbar_save.setObjectName("toolbar_save")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar_save)
        self.dock_data_inspector = QtWidgets.QDockWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dock_data_inspector.sizePolicy().hasHeightForWidth())
        self.dock_data_inspector.setSizePolicy(sizePolicy)
        self.dock_data_inspector.setMaximumSize(QtCore.QSize(524287, 524287))
        self.dock_data_inspector.setBaseSize(QtCore.QSize(300, 0))
        self.dock_data_inspector.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable|QtWidgets.QDockWidget.DockWidgetMovable)
        self.dock_data_inspector.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea|QtCore.Qt.RightDockWidgetArea)
        self.dock_data_inspector.setObjectName("dock_data_inspector")
        self.data_inspector_contents = QtWidgets.QWidget()
        self.data_inspector_contents.setObjectName("data_inspector_contents")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.data_inspector_contents)
        self.verticalLayout.setContentsMargins(1, 1, 1, 1)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tree_widget_data = QtWidgets.QTreeWidget(self.data_inspector_contents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tree_widget_data.sizePolicy().hasHeightForWidth())
        self.tree_widget_data.setSizePolicy(sizePolicy)
        self.tree_widget_data.setMinimumSize(QtCore.QSize(400, 0))
        self.tree_widget_data.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.tree_widget_data.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tree_widget_data.setIndentation(20)
        self.tree_widget_data.setUniformRowHeights(True)
        self.tree_widget_data.setAllColumnsShowFocus(True)
        self.tree_widget_data.setColumnCount(3)
        self.tree_widget_data.setObjectName("tree_widget_data")
        self.tree_widget_data.header().setDefaultSectionSize(90)
        self.tree_widget_data.header().setHighlightSections(True)
        self.verticalLayout.addWidget(self.tree_widget_data)
        self.button_load_file = QtWidgets.QPushButton(self.data_inspector_contents)
        self.button_load_file.setObjectName("button_load_file")
        self.verticalLayout.addWidget(self.button_load_file)
        self.dock_data_inspector.setWidget(self.data_inspector_contents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dock_data_inspector)
        self.action_load_new_DICOM = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/resources/database-plus.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_load_new_DICOM.setIcon(icon1)
        self.action_load_new_DICOM.setObjectName("action_load_new_DICOM")
        self.action_set_working_directory = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/resources/folder-account.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_set_working_directory.setIcon(icon2)
        self.action_set_working_directory.setObjectName("action_set_working_directory")
        self.action_save_and_propagate = QtWidgets.QAction(MainWindow)
        self.action_save_and_propagate.setEnabled(False)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/resources/progress-check.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_save_and_propagate.setIcon(icon3)
        self.action_save_and_propagate.setObjectName("action_save_and_propagate")
        self.action_crop_module = QtWidgets.QAction(MainWindow)
        self.action_crop_module.setCheckable(True)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/icons/resources/crop.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_crop_module.setIcon(icon4)
        self.action_crop_module.setObjectName("action_crop_module")
        self.action_segmentation_module = QtWidgets.QAction(MainWindow)
        self.action_segmentation_module.setCheckable(True)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/icons/resources/drawing-box.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_segmentation_module.setIcon(icon5)
        self.action_segmentation_module.setObjectName("action_segmentation_module")
        self.action_centerline_module = QtWidgets.QAction(MainWindow)
        self.action_centerline_module.setCheckable(True)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/icons/resources/vector-polyline-edit.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_centerline_module.setIcon(icon6)
        self.action_centerline_module.setObjectName("action_centerline_module")
        self.action_quit = QtWidgets.QAction(MainWindow)
        self.action_quit.setCheckable(True)
        self.action_quit.setObjectName("action_quit")
        self.action_stenosis_classifier = QtWidgets.QAction(MainWindow)
        self.action_stenosis_classifier.setCheckable(True)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/icons/resources/chart-bell-curve-cumulative.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_stenosis_classifier.setIcon(icon7)
        self.action_stenosis_classifier.setObjectName("action_stenosis_classifier")
        self.action_data_inspector = QtWidgets.QAction(MainWindow)
        self.action_data_inspector.setCheckable(True)
        self.action_data_inspector.setChecked(True)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/icons/resources/database-eye.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_data_inspector.setIcon(icon8)
        self.action_data_inspector.setObjectName("action_data_inspector")
        self.action_discard_changes = QtWidgets.QAction(MainWindow)
        self.action_discard_changes.setEnabled(False)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap(":/icons/resources/delete-forever.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_discard_changes.setIcon(icon9)
        self.action_discard_changes.setObjectName("action_discard_changes")
        self.menuFile.addAction(self.action_load_new_DICOM)
        self.menuFile.addAction(self.action_set_working_directory)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.action_save_and_propagate)
        self.menuFile.addAction(self.action_discard_changes)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.action_quit)
        self.menuView.addAction(self.action_data_inspector)
        self.menuView.addSeparator()
        self.menuView.addAction(self.action_crop_module)
        self.menuView.addAction(self.action_segmentation_module)
        self.menuView.addAction(self.action_centerline_module)
        self.menuView.addSeparator()
        self.menuView.addAction(self.action_stenosis_classifier)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.toolbar_modules.addAction(self.action_data_inspector)
        self.toolbar_modules.addSeparator()
        self.toolbar_modules.addAction(self.action_crop_module)
        self.toolbar_modules.addAction(self.action_segmentation_module)
        self.toolbar_modules.addAction(self.action_centerline_module)
        self.toolbar_modules.addSeparator()
        self.toolbar_modules.addAction(self.action_stenosis_classifier)
        self.toolbar_save.addAction(self.action_save_and_propagate)
        self.toolbar_save.addAction(self.action_discard_changes)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Carotid Analyzer"))
        self.label.setText(_translate("MainWindow", "Select a module."))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.toolbar_modules.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.toolbar_save.setWindowTitle(_translate("MainWindow", "toolbar_save"))
        self.dock_data_inspector.setWindowTitle(_translate("MainWindow", "Data Inspector"))
        self.tree_widget_data.headerItem().setText(0, _translate("MainWindow", "Patient ID"))
        self.tree_widget_data.headerItem().setText(1, _translate("MainWindow", "Left"))
        self.tree_widget_data.headerItem().setText(2, _translate("MainWindow", "Right"))
        self.button_load_file.setText(_translate("MainWindow", "Load Selected Patient"))
        self.action_load_new_DICOM.setText(_translate("MainWindow", "Load New DICOM..."))
        self.action_load_new_DICOM.setShortcut(_translate("MainWindow", "Ctrl+N"))
        self.action_set_working_directory.setText(_translate("MainWindow", "Set Working Directory..."))
        self.action_set_working_directory.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.action_save_and_propagate.setText(_translate("MainWindow", "Save And Propagate"))
        self.action_save_and_propagate.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.action_crop_module.setText(_translate("MainWindow", "Crop Module"))
        self.action_segmentation_module.setText(_translate("MainWindow", "Segmentation Module"))
        self.action_centerline_module.setText(_translate("MainWindow", "Centerline Module"))
        self.action_quit.setText(_translate("MainWindow", "Close"))
        self.action_stenosis_classifier.setText(_translate("MainWindow", "Stenosis Classifier"))
        self.action_data_inspector.setText(_translate("MainWindow", "Data Inspector"))
        self.action_discard_changes.setText(_translate("MainWindow", "Discard Changes"))
import resources_rc
