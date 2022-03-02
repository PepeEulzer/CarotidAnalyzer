import os
import sys
import glob

from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QMainWindow, QMessageBox, QTableWidgetItem
)

from mainwindow_ui import Ui_MainWindow

class CarotidAnalyzer(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # state
        self.unsaved_changes = False
        self.compute_threads_active = 0
        self.working_dir = ""
        self.patient_folder_paths = []

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
        self.action_set_working_directory.triggered.connect(self.set_working_dir)
        self.action_data_inspector.triggered[bool].connect(self.view_data_inspector)
        self.action_crop_module.triggered[bool].connect(self.view_crop_module)
        self.action_segmentation_module.triggered[bool].connect(self.view_segmentation_module)
        self.action_centerline_module.triggered[bool].connect(self.view_centerline_module)
        self.action_stenosis_classifier.triggered[bool].connect(self.view_stenosis_classifier)
        self.action_discard_changes.triggered.connect(self.discard_changes)
        self.action_save_and_propagate.triggered.connect(self.save_and_propagate)
        self.action_quit.triggered.connect(self.close)

    def load_new_DICOM(self):
        print("Call file dialog. Load a DICOM dataset")

    def set_working_dir(self):
        dir = QFileDialog.getExistingDirectory(self, "Set Working Directory")
        if len(dir) <= 0:
            return
        self.working_dir = dir
        self.patient_folder_paths = glob.glob(os.path.join(dir, "patient*"))
        self.tableWidget.clearContents()
        self.tableWidget.setRowCount(len(self.patient_folder_paths))
        for i in range(len(self.patient_folder_paths)):
            patient_ID = os.path.basename(self.patient_folder_paths[i])
            self.tableWidget.setItem(i, 0, QTableWidgetItem(patient_ID))
        self.tableWidget.resizeColumnsToContents()
            
            


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
            #settings = QSettings()
            
            # save last opened file
            #filename = QVariant(self.state.target_path) if self.state.target_path is not None else QVariant()
            #settings.setValue("LastFile", filename)
            
            # save main window position and size
            #settings.setValue("MainWindow/Geometry", QVariant(self.saveGeometry()))
            
            # save main window layout
            #settings.setValue("MainWindow/State", QVariant(self.saveState()))
            
            #self.view_3D.Finalize()
            #self.view_probe.Finalize()
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