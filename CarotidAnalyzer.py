import os
import sys
import glob

from PyQt5.QtCore import QSettings, QVariant
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QMainWindow, QMessageBox, 
    QTreeWidgetItem
)
from PyQt5.QtGui import QColor

from defaults import *
from mainwindow_ui import Ui_MainWindow
from modules.CropModule import CropModule
from modules.CenterlineModule import CenterlineModule
from modules.SegmentationModule import SegmentationModule

class CarotidAnalyzer(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # state
        self.unsaved_changes = False
        self.compute_threads_active = 0
        self.working_dir = ""
        self.patient_data = []
        self.active_patient_dict = {}
        self.active_patient_tree_widget_item = None

        # instantiate modules
        self.crop_module = CropModule(self)
        self.segmentation_module = SegmentationModule(self)
        self.centerline_module = CenterlineModule(self)
        self.module_stack.addWidget(self.crop_module)
        self.module_stack.addWidget(self.segmentation_module)
        self.module_stack.addWidget(self.centerline_module)

        # only one module can be active
        self.processing_modules = [
            self.action_crop_module, self.action_segmentation_module, self.action_centerline_module
        ]
        self.vis_modules = [
            self.action_stenosis_classifier
        ]
        
        # connect signals to slots
        self.action_load_new_DICOM.triggered.connect(self.loadNewDICOM)
        self.action_set_working_directory.triggered.connect(self.openWorkingDirDialog)
        self.action_data_inspector.triggered[bool].connect(self.viewDataInspector)
        self.action_crop_module.triggered[bool].connect(self.viewCropModule)
        self.action_segmentation_module.triggered[bool].connect(self.viewSegmentationModule)
        self.action_centerline_module.triggered[bool].connect(self.viewCenterlineModule)
        self.action_stenosis_classifier.triggered[bool].connect(self.viewStenosisClassifier)
        self.action_discard_changes.triggered.connect(self.discardChanges)
        self.action_save_and_propagate.triggered.connect(self.saveAndPropagate)
        self.action_quit.triggered.connect(self.close)
        self.button_load_file.clicked.connect(self.loadSelectedPatient)

        self.centerline_module.centerline_module_left.data_modified.connect(self.changesMade)
        self.centerline_module.centerline_module_right.data_modified.connect(self.changesMade)
        self.centerline_module.new_centerlines.connect(self.newCenterlines)
        self.segmentation_module.segmentation_module_left.data_modified.connect(self.changesMade)
        self.segmentation_module.segmentation_module_right.data_modified.connect(self.changesMade)
        self.segmentation_module.new_segmentation.connect(self.newSegmentation)
        self.segmentation_module.new_models.connect(self.newModels)


        # restore state properties
        settings = QSettings()
        # geometry = settings.value("MainWindow/Geometry")
        # if geometry:
        #     self.restoreGeometry(geometry)
            
        dir = settings.value("LastWorkingDir")
        if dir != None:
            self.setWorkingDir(dir)


    def loadNewDICOM(self):
        print("Call file dialog. Load a DICOM dataset")
        print("NOT IMPLEMENTED")

    
    def setWorkingDir(self, dir):
        if len(dir) <= 0:
            return
        self.working_dir = dir
        self.patient_data = []
        self.tree_widget_data.clear()

        for patient_folder in glob.glob(os.path.join(dir, "patient*")):
            pID = os.path.basename(patient_folder)
            patient_dict = {}
            patient_dict['patient_ID'] = pID
            patient_dict['base_path'] = patient_folder

            # create a models subdir if it does not exist
            model_path = os.path.join(patient_folder, "models")
            if not os.path.exists(model_path):
                os.makedirs(model_path)

            def add_if_exists(dict_key, file_tail, models_subdir=False):
                if models_subdir:
                    path = os.path.join(patient_folder, "models", pID + file_tail)
                else:
                    path = os.path.join(patient_folder, pID + file_tail)
                if os.path.exists(path):
                    patient_dict[dict_key] = path
                else:
                    patient_dict[dict_key] = False

            # Fills the patient dict with all existing filepaths.
            # Non-existing filepaths are marked with 'False'.
            # First param is the dict key used to retrieve the entry.
            # Second param is the file tail after patientID.
            # If the third param is true, the 'models' subdirectory is used.
            add_if_exists("volume_raw", ".nrrd")
            add_if_exists("volume_left", "_left.nrrd")
            add_if_exists("volume_right", "_right.nrrd")
            add_if_exists("seg_left", "_left.seg.nrrd")
            add_if_exists("seg_left_pred", "_left_pred.seg.nrrd")
            add_if_exists("seg_right", "_right.seg.nrrd")
            add_if_exists("seg_right_pred", "_right_pred.seg.nrrd")
            add_if_exists("lumen_model_left", "_left_lumen.stl", True)
            add_if_exists("lumen_model_right", "_right_lumen.stl", True)
            add_if_exists("plaque_model_left", "_left_plaque.stl", True)
            add_if_exists("plaque_model_right", "_right_plaque.stl", True)
            add_if_exists("centerlines_left", "_left_lumen_centerlines.vtp", True)
            add_if_exists("centerlines_right", "_right_lumen_centerlines.vtp", True)
            self.patient_data.append(patient_dict)

            entry_volume = ["Crop Volume", "", ""]
            entry_volume[1] = SYM_YES if patient_dict["volume_left"] else SYM_NO
            entry_volume[2] = SYM_YES if patient_dict["volume_right"] else SYM_NO

            entry_seg = ["Segmentation", "", ""]
            entry_seg[1] = SYM_YES if patient_dict["seg_left"] else SYM_NO
            entry_seg[2] = SYM_YES if patient_dict["seg_right"] else SYM_NO

            entry_lumen = ["Lumen Model", "", ""]
            entry_lumen[1] = SYM_YES if patient_dict["lumen_model_left"] else SYM_NO
            entry_lumen[2] = SYM_YES if patient_dict["lumen_model_right"] else SYM_NO

            entry_plaque = ["Plaque Model", "", ""]
            entry_plaque[1] = SYM_YES if patient_dict["plaque_model_left"] else SYM_NO
            entry_plaque[2] = SYM_YES if patient_dict["plaque_model_right"] else SYM_NO

            entry_centerlines = ["Centerlines", "", ""]
            entry_centerlines[1] = SYM_YES if patient_dict["centerlines_left"] else SYM_NO
            entry_centerlines[2] = SYM_YES if patient_dict["centerlines_right"] else SYM_NO

            entry_patient = QTreeWidgetItem([pID, "", ""])
            background_color = QColor(240, 240, 240)
            entry_patient.setBackground(0, background_color)
            entry_patient.setBackground(1, background_color)
            entry_patient.setBackground(2, background_color)
            entry_patient.addChild(QTreeWidgetItem(entry_volume))
            entry_patient.addChild(QTreeWidgetItem(entry_seg))
            entry_patient.addChild(QTreeWidgetItem(entry_lumen))
            entry_patient.addChild(QTreeWidgetItem(entry_plaque))
            entry_patient.addChild(QTreeWidgetItem(entry_centerlines))
            self.tree_widget_data.addTopLevelItem(entry_patient)
            entry_patient.setExpanded(True)
        self.tree_widget_data.resizeColumnToContents(0)
        self.tree_widget_data.resizeColumnToContents(1)
        self.tree_widget_data.resizeColumnToContents(2)
        
    
    def openWorkingDirDialog(self):
        dir = QFileDialog.getExistingDirectory(self, "Set Working Directory")
        if len(dir) > 0:
            self.setWorkingDir(dir)


    def loadSelectedPatient(self):
        selected = self.tree_widget_data.currentItem()
        while selected.parent() != None:
            selected = selected.parent()
        self.active_patient_tree_widget_item = selected
        patient_ID = selected.text(0)
        for patient in self.patient_data:
            if (patient['patient_ID'] == patient_ID):
                # update patient in all modules
                self.active_patient_dict = patient
                self.crop_module.loadPatient(patient)
                self.segmentation_module.loadPatient(patient)
                self.centerline_module.loadPatient(patient)
                self.statusbar.showMessage(patient['patient_ID'])
                break

    
    def viewDataInspector(self, on:bool):
        if on:
            self.dock_data_inspector.show()
        else:
            self.dock_data_inspector.close()

    
    def uncheckInactiveModules(self, active_module):
        for m in self.processing_modules + self.vis_modules:
            if not m == active_module:
                m.setChecked(False)

    
    def viewCropModule(self, on:bool):
        if on:
            self.uncheckInactiveModules(self.action_crop_module)
            self.module_stack.setCurrentWidget(self.crop_module)
        else:
            self.module_stack.setCurrentWidget(self.empty_module)

    
    def viewSegmentationModule(self, on:bool):
        if on:
            self.uncheckInactiveModules(self.action_segmentation_module)
            self.module_stack.setCurrentWidget(self.segmentation_module)
        else:
            self.module_stack.setCurrentWidget(self.empty_module)

    
    def viewCenterlineModule(self, on:bool):
        if on:
            self.uncheckInactiveModules(self.action_centerline_module)
            self.module_stack.setCurrentWidget(self.centerline_module)
        else:
            self.module_stack.setCurrentWidget(self.empty_module)

    
    def viewStenosisClassifier(self, on:bool):
        if on:
            self.uncheckInactiveModules(self.action_stenosis_classifier)
        else:
            self.module_stack.setCurrentWidget(self.empty_module)


    def changesMade(self):
        self.unsaved_changes = True
        self.action_discard_changes.setEnabled(True)
        self.action_save_and_propagate.setEnabled(True)
    

    def discardChanges(self):
        self.module_stack.currentWidget().discard()
        self.action_discard_changes.setEnabled(False)
        self.action_save_and_propagate.setEnabled(False)
        self.unsaved_changes = False

    
    def saveAndPropagate(self):
        # calls save on the current widget
        # propagation must be called through widget signals of type "newX"
        self.module_stack.currentWidget().save()
        self.action_discard_changes.setEnabled(False)
        self.action_save_and_propagate.setEnabled(False)
        self.unsaved_changes = False


    def newSegmentation(self):
        patient_ID = self.active_patient_dict['patient_ID']
        base_path  = self.active_patient_dict['base_path']
        path_left = os.path.join(base_path, patient_ID + "_left.seg.nrrd")
        path_right = os.path.join(base_path, patient_ID + "_right.seg.nrrd")
        seg_item = self.active_patient_tree_widget_item.child(1)
        if os.path.exists(path_left):
            self.active_patient_dict['seg_left'] = path_left
            seg_item.setText(1, SYM_YES)
        if os.path.exists(path_right):
            self.active_patient_dict['seg_right'] = path_right
            seg_item.setText(2, SYM_YES)

    
    def newModels(self):
        patient_ID = self.active_patient_dict['patient_ID']
        base_path  = self.active_patient_dict['base_path']
        path_left_lumen = os.path.join(base_path, "models", patient_ID + "_left_lumen.stl")
        path_left_plaque = os.path.join(base_path, "models", patient_ID + "_left_plaque.stl")
        path_right_lumen = os.path.join(base_path, "models", patient_ID + "_right_lumen.stl")
        path_right_plaque = os.path.join(base_path, "models", patient_ID + "_right_plaque.stl")
        lumen_item = self.active_patient_tree_widget_item.child(2)
        plaque_item = self.active_patient_tree_widget_item.child(3)
        if os.path.exists(path_left_lumen):
            self.active_patient_dict['lumen_model_left'] = path_left_lumen
            lumen_item.setText(1, SYM_YES)
        if os.path.exists(path_left_plaque):
            self.active_patient_dict['plaque_model_left'] = path_left_plaque
            plaque_item.setText(1, SYM_YES)
        if os.path.exists(path_right_lumen):
            self.active_patient_dict['lumen_model_right'] = path_right_lumen
            lumen_item.setText(2, SYM_YES)
        if os.path.exists(path_right_plaque):
            self.active_patient_dict['plaque_model_right'] = path_right_plaque
            plaque_item.setText(2, SYM_YES)

        # propagate
        self.centerline_module.loadPatient(self.active_patient_dict)

    
    def newCenterlines(self):
        patient_ID = self.active_patient_dict['patient_ID']
        base_path  = self.active_patient_dict['base_path']
        path_left = os.path.join(base_path, "models", patient_ID + "_left_lumen_centerlines.vtp")
        path_right = os.path.join(base_path, "models", patient_ID + "_right_lumen_centerlines.vtp")
        centerlines_item = self.active_patient_tree_widget_item.child(4)
        if os.path.exists(path_left):
            self.active_patient_dict['centerlines_left'] = path_left
            centerlines_item.setText(1, SYM_YES)
        if os.path.exists(path_right):
            self.active_patient_dict['centerlines_right'] = path_right
            centerlines_item.setText(2, SYM_YES)


    def okToClose(self):
        if self.unsaved_changes:
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Unsaved Changes")
            dlg.setText("Close application? Changes will be lost.")
            dlg.setStandardButtons(QMessageBox.Close | QMessageBox.Cancel)
            button = dlg.exec()
            if button == QMessageBox.Cancel:
                return False

        if self.compute_threads_active > 0:
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Threads Running")
            dlg.setText("Close application? Active computations will be lost.")
            dlg.setStandardButtons(QMessageBox.Close | QMessageBox.Cancel)
            button = dlg.exec()
            if button == QMessageBox.Cancel:
                return False

        return True
    
    
    def closeEvent(self, event):
        if self.okToClose():
            settings = QSettings()
            
            # save last opened working dir
            dirname = QVariant(self.working_dir) #if len(self.working_dir) is not 0 else QVariant()
            settings.setValue("LastWorkingDir", dirname)
            
            # save main window position and size
            settings.setValue("MainWindow/Geometry", QVariant(self.saveGeometry()))
            
            # call Finalize() for all vtk interactors
            self.crop_module.close()
            self.segmentation_module.close()
            self.centerline_module.close()
            super(CarotidAnalyzer, self).closeEvent(event)
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setOrganizationName("VisGroup Uni Jena")
    app.setOrganizationDomain("vis.uni-jena.de")
    app.setApplicationName("CarotidAnalyzer")
    app.setStyle("Fusion")
    win = CarotidAnalyzer()
    win.show()
    sys.exit(app.exec())