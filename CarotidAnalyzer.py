from email.mime import base
import os
import sys
import glob
from xml.etree.ElementTree import PI

from PyQt5.QtCore import QSettings, QVariant
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QMainWindow, QMessageBox, QTreeWidgetItem
)
from PyQt5.QtGui import QColor

from mainwindow_ui import Ui_MainWindow

class CarotidAnalyzer(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # state
        self.unsaved_changes = False
        self.compute_threads_active = 0
        self.working_dir = ""
        self.patient_data = []

        # only one module can be active
        self.active_module = None
        self.processing_modules = [
            self.action_crop_module, self.action_segmentation_module, self.action_centerline_module
        ]
        self.vis_modules = [
            self.action_stenosis_classifier
        ]
        
        # connect signals to slots
        self.action_load_new_DICOM.triggered.connect(self.load_new_DICOM)
        self.action_set_working_directory.triggered.connect(self.open_working_dir_dialog)
        self.action_data_inspector.triggered[bool].connect(self.view_data_inspector)
        self.action_crop_module.triggered[bool].connect(self.view_crop_module)
        self.action_segmentation_module.triggered[bool].connect(self.view_segmentation_module)
        self.action_centerline_module.triggered[bool].connect(self.view_centerline_module)
        self.action_stenosis_classifier.triggered[bool].connect(self.view_stenosis_classifier)
        self.action_discard_changes.triggered.connect(self.discard_changes)
        self.action_save_and_propagate.triggered.connect(self.save_and_propagate)
        self.action_quit.triggered.connect(self.close)

        # restore state properties
        settings = QSettings()
        # geometry = settings.value("MainWindow/Geometry")
        # if geometry:
        #     self.restoreGeometry(geometry)
            
        dir = settings.value("LastWorkingDir")
        if dir != None:
            self.set_working_dir(dir)


    def load_new_DICOM(self):
        print("Call file dialog. Load a DICOM dataset")

    
    def set_working_dir(self, dir):
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
            add_if_exists("seg_right", "_right.seg.nrrd")
            add_if_exists("lumen_model_left", "_left_lumen.stl", True)
            add_if_exists("lumen_model_right", "_right_lumen.stl", True)
            add_if_exists("plaque_model_left", "_left_plaque.stl", True)
            add_if_exists("plaque_model_right", "_right_plaque.stl", True)
            add_if_exists("centerlines_left", "_left_lumen_centerlines.vtp", True)
            add_if_exists("centerlines_right", "_right_lumen_centerlines.vtp", True)
            self.patient_data.append(patient_dict)

            # unicode symbols
            sym_yes = "\u2714"
            sym_no = "\u2716"
            #sym_endash = "\u2013"
            
            entry_volume = ["CTA Volume", "", ""]
            entry_volume[1] = sym_yes if patient_dict["volume_left"] else sym_no
            entry_volume[2] = sym_yes if patient_dict["volume_right"] else sym_no

            entry_seg = ["Segmentation", "", ""]
            entry_seg[1] = sym_yes if patient_dict["seg_left"] else sym_no
            entry_seg[2] = sym_yes if patient_dict["seg_right"] else sym_no

            entry_lumen = ["Lumen Model", "", ""]
            entry_lumen[1] = sym_yes if patient_dict["lumen_model_left"] else sym_no
            entry_lumen[2] = sym_yes if patient_dict["lumen_model_right"] else sym_no

            entry_plaque = ["Plaque Model", "", ""]
            entry_plaque[1] = sym_yes if patient_dict["plaque_model_left"] else sym_no
            entry_plaque[2] = sym_yes if patient_dict["plaque_model_right"] else sym_no

            entry_centerlines = ["Centerlines", "", ""]
            entry_centerlines[1] = sym_yes if patient_dict["centerlines_left"] else sym_no
            entry_centerlines[2] = sym_yes if patient_dict["centerlines_right"] else sym_no

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
        
    
    def open_working_dir_dialog(self):
        dir = QFileDialog.getExistingDirectory(self, "Set Working Directory")
        if len(dir) > 0:
            self.set_working_dir(dir)

    
    def view_data_inspector(self, on:bool):
        if on:
            self.dock_data_inspector.show()
        else:
            self.dock_data_inspector.close()

    
    def activate_module(self, module):
        for m in self.processing_modules + self.vis_modules:
            if m == module:
                print("Activating module", m.objectName())
                self.active_module = m
            else:
                m.setChecked(False)

    
    def view_crop_module(self, on:bool):
        if on:
            self.activate_module(self.action_crop_module)
        else:
            print("Deactivating module", self.action_crop_module.objectName())
            self.active_module = None
    
    
    def view_segmentation_module(self, on:bool):
        if on:
            self.activate_module(self.action_segmentation_module)
        else:
            print("Deactivating module", self.action_segmentation_module.objectName())
            self.active_module = None

    
    def view_centerline_module(self, on:bool):
        if on:
            self.activate_module(self.action_centerline_module)
        else:
            print("Deactivating module", self.action_centerline_module.objectName())
            self.active_module = None

    
    def view_stenosis_classifier(self, on:bool):
        if on:
            self.activate_module(self.action_stenosis_classifier)
        else:
            print("Deactivating module", self.action_stenosis_classifier.objectName())
            self.active_module = None

    
    def discard_changes(self):
        print("Call discard function of active module.")
        self.action_discard_changes.setEnabled(False)
        self.action_save_and_propagate.setEnabled(False)

    
    def save_and_propagate(self):
        print("Call save function of active module. Update following steps.")
        self.action_discard_changes.setEnabled(False)
        self.action_save_and_propagate.setEnabled(False)

    
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
            
            # call view.Finalize() for all vtk views here
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