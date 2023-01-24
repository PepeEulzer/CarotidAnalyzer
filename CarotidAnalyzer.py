import os
import sys
import shutil
import glob
from collections import OrderedDict

import numpy as np 
import nrrd
import pydicom
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from PyQt5.QtCore import QSettings, QVariant, QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QMainWindow, QMessageBox, 
    QTreeWidgetItem, QInputDialog, QProgressBar
)
from PyQt5.QtGui import QColor

from defaults import *
from mainwindow_ui import Ui_MainWindow
from modules.CropModule import CropModule
from modules.CenterlineModule import CenterlineModule
from modules.SegmentationModule import SegmentationModule
from modules.StenosisClassifier import StenosisClassifier

class CarotidAnalyzer(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.tree_widget_data.setExpandsOnDoubleClick(False)

        # state
        self.unsaved_changes = False
        self.compute_threads_active = 0
        self.working_dir = ""
        self.patient_data = []
        self.active_patient_dict = {}
        self.active_patient_tree_widget_item = None
        self.data = []
        self.locations = []
        self.DICOM_source_dir = ""
        self.DICOM_patient_ID = ""

        # instantiate modules
        self.crop_module = CropModule(self)
        self.segmentation_module = SegmentationModule(self)
        self.centerline_module = CenterlineModule(self)
        self.stenosis_classifier = StenosisClassifier(self)
        self.module_stack.addWidget(self.crop_module)
        self.module_stack.addWidget(self.segmentation_module)
        self.module_stack.addWidget(self.centerline_module)
        self.module_stack.addWidget(self.stenosis_classifier)

        # only one module can be active
        self.processing_modules = [
            self.action_crop_module, self.action_segmentation_module, self.action_centerline_module
        ]
        self.vis_modules = [
            self.action_stenosis_classifier
        ]
        
        # connect signals to slots
        self.action_load_new_DICOM.triggered.connect(self.openDICOMDirDialog)
        self.action_set_working_directory.triggered.connect(self.openWorkingDirDialog)
        self.action_delete_selected_patient.triggered.connect(self.deleteSelectedPatient)
        self.action_data_inspector.triggered[bool].connect(self.viewDataInspector)
        self.action_crop_module.triggered[bool].connect(self.viewCropModule)
        self.action_segmentation_module.triggered[bool].connect(self.viewSegmentationModule)
        self.action_centerline_module.triggered[bool].connect(self.viewCenterlineModule)
        self.action_stenosis_classifier.triggered[bool].connect(self.viewStenosisClassifier)
        self.action_discard_changes.triggered.connect(self.discardChanges)
        self.action_save_and_propagate.triggered.connect(self.saveAndPropagate)
        self.action_quit.triggered.connect(self.close)
        self.button_load_file.clicked.connect(self.loadSelectedPatient)
        self.tree_widget_data.itemDoubleClicked.connect(self.loadSelectedPatient)

        self.crop_module.data_modified.connect(self.changesMade)
        self.crop_module.new_left_volume.connect(self.newLeftVolume)
        self.crop_module.new_right_volume.connect(self.newRightVolume)
        self.segmentation_module.segmentation_module_left.data_modified.connect(self.changesMade)
        self.segmentation_module.segmentation_module_right.data_modified.connect(self.changesMade)
        self.segmentation_module.new_segmentation.connect(self.newSegmentation)
        self.segmentation_module.new_models.connect(self.newModels)
        self.centerline_module.centerline_module_left.data_modified.connect(self.changesMade)
        self.centerline_module.centerline_module_right.data_modified.connect(self.changesMade)
        self.centerline_module.new_centerlines.connect(self.newCenterlines)

        # restore state properties
        settings = QSettings()
        # geometry = settings.value("MainWindow/Geometry")
        # if geometry:
        #     self.restoreGeometry(geometry)
        dir = settings.value("LastWorkingDir")
        if dir != None:
            self.setWorkingDir(dir)
 
    
    def loadNewDICOM(self, data_array): 
        # get metadata for header/vtkImage
        dicomdata = pydicom.dcmread(os.path.join(self.DICOM_source_dir, os.listdir(self.DICOM_source_dir)[0]))
        dim_x, dim_y, dim_z = data_array.shape
        s_z = float(dicomdata[0x0018, 0x0088].value)  # spacing between slices 
        s_x_y = dicomdata[0x0028, 0x0030].value  # pixel spacing 
        pos = dicomdata[0x0020, 0x0032].value  # image position

        # user input if dicom data should be saved in nrrd
        save_nrrd = QMessageBox.question(self, "Save Full Volume", "Should the full volume be saved in a .nrrd file?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if save_nrrd == QMessageBox.Yes:
            # save as nrrd 
            filename = self.DICOM_patient_ID + ".nrrd"
            nrrd_path = os.path.join(self.working_dir, self.DICOM_patient_ID, filename)
            header = OrderedDict()
            header['dimension'] = 3
            header['space'] = 'left-posterior-superior'
            header['sizes'] =  str(dim_x) + ' ' + str(dim_y) + ' ' + str(dim_z) 
            header['space directions'] = [[s_x_y[0], 0.0, 0.0], [0.0, s_x_y[1], 0.0], [0.0, 0.0, s_z]]
            header['kinds'] = ['domain', 'domain', 'domain']
            header['endian'] = 'little'
            header['encoding'] = 'gzip'
            header['space origin'] = pos
            self.write_nrrd(nrrd_path, data_array, header, filename)
           
        # convert to vtkImage
        image = vtk.vtkImageData()
        image.SetDimensions(dim_x,dim_y,dim_z)
        image.SetSpacing(s_x_y[0], s_x_y[1], s_z)
        image.SetOrigin(pos)
        vtk_data_array = numpy_to_vtk(data_array.ravel(order='F'))
        image.GetPointData().SetScalars(vtk_data_array)

        # update tree widget, load patient
        self.setWorkingDir(self.working_dir)   
        for patient in self.patient_data:
            if patient['patient_ID'] == self.DICOM_patient_ID:
                self.active_patient_dict = patient
                self.crop_module.loadPatient(patient, image)
                self.segmentation_module.loadPatient(patient)
                self.centerline_module.loadPatient(patient)
                self.stenosis_classifier.loadPatient(patient)
                break
        
        # set as activated widget
        for i in range(self.tree_widget_data.topLevelItemCount()):
                if self.tree_widget_data.topLevelItem(i).text(0) == self.DICOM_patient_ID:
                    self.active_patient_tree_widget_item = self.tree_widget_data.topLevelItem(i)
                    break
        self.setPatientTreeItemColor(self.active_patient_tree_widget_item, COLOR_SELECTED)


    def write_nrrd(self, path, array, header, filename):
        self.button_load_file.setEnabled(False)
        self.statusbar.showMessage("Saving "+filename+" ...")

        self.thread = QThread()
        self.worker = NrrdWriterWorker()
        self.worker.moveToThread(self.thread)
        self.worker.path = path 
        self.worker.array = array
        self.worker.header = header
        self.thread.started.connect(self.worker.run) 
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        self.thread.finished.connect(lambda: self.button_load_file.setEnabled(True))
        self.thread.finished.connect(lambda: self.statusbar.clearMessage())

    
    def report_DICOM_Progress(self, progress_val, progress_msg):
        self.pbar.setValue(progress_val)
        self.pbar.setFormat(progress_msg + " (%p%)")
        

    def openDICOMDirDialog(self): 
        # set path for dcm file
        source_dir = QFileDialog.getExistingDirectory(self, "Set source directory of DICOM files")
        
        # userinput for target filename if dcm path set 
        if source_dir:
            dir_name, ok = QInputDialog.getText(self, "Set Patient Directory", "Enter name of directory for patient data:")
            # check if directory exists, if yes -> open new dialog and check again 
            if dir_name and ok:
                while (os.path.exists(os.path.join(self.working_dir, dir_name)) or
                       os.path.exists(os.path.join(self.working_dir,("patient_" + dir_name)))):
                    dir_name, ok = QInputDialog.getText(self, "Set patient Directory", "Directory/Patient allready exists! Please choose another name:")
                    # break if dialog canceled by user 
                    if not ok: 
                        break

                if dir_name and ok: 
                    # check if name correct so that data can be found later
                    if not dir_name.startswith('patient'):
                        dir_name = "patient_" + dir_name
                    
                    # create directory 
                    path = os.path.join(self.working_dir, dir_name) 
                    os.mkdir(path)
                    os.mkdir(os.path.join(path, "models"))
                    self.DICOM_source_dir = source_dir
                    self.DICOM_patient_ID = dir_name
                    
                    # start new thread to read dicom and report progress
                    self.action_load_new_DICOM.setEnabled(False)
                    self.pbar = QProgressBar() 
                    self.pbar.setMinimum(0)
                    self.pbar.setMaximum(len(os.listdir(source_dir)) + 1)
                    self.statusbar.addWidget(self.pbar)

                    self.thread = QThread()
                    self.worker = DICOMReaderWorker()
                    self.worker.source_dir = source_dir
                    self.worker.moveToThread(self.thread)

                    self.worker.progress[int, str].connect(self.report_DICOM_Progress)  
                    self.worker.data_processed[object].connect(self.loadNewDICOM)
                    self.worker.finished.connect(self.thread.quit)
                    self.worker.finished.connect(self.worker.deleteLater)

                    self.thread.started.connect(self.worker.run)
                    self.thread.finished.connect(lambda: self.action_load_new_DICOM.setEnabled(True))
                    self.thread.finished.connect(lambda: self.statusbar.removeWidget(self.pbar))
                    self.thread.finished.connect(self.thread.deleteLater)
                    self.thread.start()
                
                
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
            add_if_exists("seg_right", "_right.seg.nrrd")
            add_if_exists("lumen_model_left", "_left_lumen.stl", True)
            add_if_exists("lumen_model_right", "_right_lumen.stl", True)
            add_if_exists("plaque_model_left", "_left_plaque.stl", True)
            add_if_exists("plaque_model_right", "_right_plaque.stl", True)
            add_if_exists("centerlines_left", "_left_lumen_centerlines.vtp", True)
            add_if_exists("centerlines_right", "_right_lumen_centerlines.vtp", True)
            self.patient_data.append(patient_dict)

            entry_volume_raw = ["Full Volume", "", ""]
            entry_volume_raw[1] = SYM_YES if patient_dict["volume_raw"] else SYM_NO
            entry_volume_raw[2] = entry_volume_raw[1]

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
            entry_patient.addChild(QTreeWidgetItem(entry_volume_raw))
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

        if not EXPAND_PATIENTS:
            for i in range(self.tree_widget_data.topLevelItemCount()):
                self.tree_widget_data.topLevelItem(i).setExpanded(False)
        
    
    def openWorkingDirDialog(self):
        dir = QFileDialog.getExistingDirectory(self, "Set Working Directory")
        if len(dir) > 0:
            self.setWorkingDir(dir)


    def setPatientTreeItemColor(self, item, color):
        if item is None:
            return
        c = QColor(color[0], color[1], color[2])
        for i in range(3):
            item.setBackground(i, c)
        for i in range(item.childCount()):
            for j in range(3):
                item.child(i).setBackground(j, c)
       

    def loadSelectedPatient(self):  
        if self.unsaved_changes:
            return

        # get top parent item
        selected = self.tree_widget_data.currentItem()
        if selected == None:
            return
            
        while selected.parent() != None:
            selected = selected.parent()

        # set colors of last and current item
        self.setPatientTreeItemColor(self.active_patient_tree_widget_item, COLOR_UNSELECTED)
        self.setPatientTreeItemColor(selected, COLOR_SELECTED)

        # save selected item, load new patient
        self.active_patient_tree_widget_item = selected
        patient_ID = selected.text(0)
        for patient in self.patient_data:
            if (patient['patient_ID'] == patient_ID):
                # update patient in all modules
                self.active_patient_dict = patient
                self.__updatePatientInModules()
                if SHOW_MODEL_MISMATCH_WARNING:
                    self.__checkSegMatchesModels()
                break


    def deleteSelectedPatient(self):
        # get selected top parent item
        selected = self.tree_widget_data.currentItem()
        if selected == None:
            return

        while selected.parent() != None: 
            selected = selected.parent()

        # delete patient dierectory and remove patient from tree widget if user confirms patient
        patient = selected.text(0)
        delete = QMessageBox.question(self,
                                      "Delete patient",
                                      "Do you want to delete the data of " + patient + "?",
                                      QMessageBox.Yes | QMessageBox.No,
                                      QMessageBox.No)
        if delete == QMessageBox.Yes:
            patient_idx = self.tree_widget_data.indexOfTopLevelItem(selected)
            shutil.rmtree(os.path.join(self.working_dir, patient))
            del self.patient_data[patient_idx]
            self.tree_widget_data.takeTopLevelItem(patient_idx)

    
    def __updatePatientInModules(self):
        self.crop_module.loadPatient(self.active_patient_dict)
        self.segmentation_module.loadPatient(self.active_patient_dict)
        self.centerline_module.loadPatient(self.active_patient_dict)
        self.stenosis_classifier.loadPatient(self.active_patient_dict)
        

    def __checkSegMatchesModels(self):
        """
        Test if a new patient's segmentation matches the model files.
        They may be out of sync if the segmentation was externally modified.
        """
        seg_model_left = self.segmentation_module.segmentation_module_left.model_view.smoother_lumen.GetOutput()
        cen_model_left = self.centerline_module.centerline_module_left.reader_lumen.GetOutput()

        seg_model_right = self.segmentation_module.segmentation_module_right.model_view.smoother_lumen.GetOutput()
        cen_model_right = self.centerline_module.centerline_module_right.reader_lumen.GetOutput()

        match = True
        for models in [[seg_model_left, cen_model_left], [seg_model_right, cen_model_right]]:
            p0 = models[0].GetNumberOfPoints()
            p1 = models[1].GetNumberOfPoints()
            if abs(p0 - p1) > 10:
                match = False
                break

            center0 = models[0].GetCenter()
            center1 = models[1].GetCenter()
            distance = abs(center0[0] - center1[0] +
                           center0[1] - center1[1] +
                           center0[2] - center1[2])
            if distance > 0.01:
                match = False
                break
        
        if not match:
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Warning: Models do not Match")
            dlg.setText("The segmentation and model files of this patient do not match. " + 
                        "This is probably due to external modifications to the segmentation. " +
                        "Do you want to overwrite the model files?")
            dlg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            button = dlg.exec()
            if button == QMessageBox.Ok:
                print("Overwriting model files...")
                self.segmentation_module.save()


    
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
            self.module_stack.setCurrentWidget(self.stenosis_classifier)
        else:
            self.module_stack.setCurrentWidget(self.empty_module)

    
    def setModulesClickable(self, state:bool):
        self.action_crop_module.setEnabled(state)
        self.action_segmentation_module.setEnabled(state)
        self.action_centerline_module.setEnabled(state)
        self.action_stenosis_classifier.setEnabled(state)


    def changesMade(self):
        self.unsaved_changes = True
        self.setModulesClickable(False)
        self.button_load_file.setEnabled(False)
        self.action_discard_changes.setEnabled(True)
        self.action_save_and_propagate.setEnabled(True)
    

    def discardChanges(self):
        self.module_stack.currentWidget().discard()
        self.action_discard_changes.setEnabled(False)
        self.action_save_and_propagate.setEnabled(False)
        self.button_load_file.setEnabled(True)
        self.unsaved_changes = False
        self.setModulesClickable(True)

    
    def saveAndPropagate(self):
        # calls save on the current widget
        # propagation must be called through widget signals of type "newX"
        self.module_stack.currentWidget().save()
        self.action_discard_changes.setEnabled(False)
        self.action_save_and_propagate.setEnabled(False)
        self.button_load_file.setEnabled(True)
        self.unsaved_changes = False
        self.setModulesClickable(True)

    
    def newLeftVolume(self):
        patient_ID = self.active_patient_dict['patient_ID']
        base_path  = self.active_patient_dict['base_path']
        path_left = os.path.join(base_path, patient_ID + "_left.nrrd")
        seg_item = self.active_patient_tree_widget_item.child(1)
        if os.path.exists(path_left):
            self.active_patient_dict['volume_left'] = path_left
            seg_item.setText(1, SYM_YES)

        # propagate
        self.segmentation_module.patient_dict = self.active_patient_dict
        self.segmentation_module.segmentation_module_left.loadVolumeSeg(
            self.active_patient_dict['volume_left'],
            self.active_patient_dict['seg_left']
        )


    def newRightVolume(self):
        patient_ID = self.active_patient_dict['patient_ID']
        base_path  = self.active_patient_dict['base_path']
        path_right = os.path.join(base_path, patient_ID + "_right.nrrd")
        seg_item = self.active_patient_tree_widget_item.child(1)
        if os.path.exists(path_right):
            self.active_patient_dict['volume_right'] = path_right
            seg_item.setText(2, SYM_YES)

        # propagate
        self.segmentation_module.patient_dict = self.active_patient_dict
        self.segmentation_module.segmentation_module_right.loadVolumeSeg(
            self.active_patient_dict['volume_right'],
            self.active_patient_dict['seg_right']
        )


    def newSegmentation(self):
        patient_ID = self.active_patient_dict['patient_ID']
        base_path  = self.active_patient_dict['base_path']
        path_left = os.path.join(base_path, patient_ID + "_left.seg.nrrd")
        path_right = os.path.join(base_path, patient_ID + "_right.seg.nrrd")
        seg_item = self.active_patient_tree_widget_item.child(2)
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
        lumen_item = self.active_patient_tree_widget_item.child(3)
        plaque_item = self.active_patient_tree_widget_item.child(4)
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
        self.stenosis_classifier.loadPatient(self.active_patient_dict)

    
    def newCenterlines(self):
        patient_ID = self.active_patient_dict['patient_ID']
        base_path  = self.active_patient_dict['base_path']
        path_left = os.path.join(base_path, "models", patient_ID + "_left_lumen_centerlines.vtp")
        path_right = os.path.join(base_path, "models", patient_ID + "_right_lumen_centerlines.vtp")
        centerlines_item = self.active_patient_tree_widget_item.child(5)
        if os.path.exists(path_left):
            self.active_patient_dict['centerlines_left'] = path_left
            centerlines_item.setText(1, SYM_YES)
        if os.path.exists(path_right):
            self.active_patient_dict['centerlines_right'] = path_right
            centerlines_item.setText(2, SYM_YES)

        # delete meta information files for stenoses if they exist
        meta_path_left = os.path.join(base_path, "models", patient_ID + "_left_meta.txt")
        meta_path_right = os.path.join(base_path, "models", patient_ID + "_right_meta.txt")
        if os.path.exists(meta_path_left):
            os.remove(meta_path_left)
        if os.path.exists(meta_path_right):
            os.remove(meta_path_right)

        # propagate
        self.stenosis_classifier.loadPatient(self.active_patient_dict)


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
            self.stenosis_classifier.close()
            super(CarotidAnalyzer, self).closeEvent(event)
        else:
            event.ignore()



class DICOMReaderWorker(QObject):  
    finished = pyqtSignal()
    progress = pyqtSignal(int, str)
    data_processed = pyqtSignal(object)
    source_dir = None 
   

    def run(self):
        data = []
        locations = []
        path = os.listdir(self.source_dir)
        for idx, file in enumerate(path):
            # read in each dcm and save pixel data, emit progress and data when finished 
            self.progress.emit(idx, "Loading " + file)
            ds = pydicom.dcmread(os.path.join(self.source_dir, file))
            hu = pydicom.pixel_data_handlers.util.apply_modality_lut(ds.pixel_array, ds)
            locations.append(ds[0x0020, 0x1041].value) # slice location
            data.append(hu)

        # sort slices if required
        self.progress.emit(idx + 1, "Sorting slices...")
        if not (all(locations[i] <= locations[i + 1] for i in range(len(locations)-1))):
            data = [x for _, x in sorted(zip(locations, data))] 
        data_array = np.transpose(np.array(data, dtype=np.int16))

        self.data_processed.emit(data_array)
        self.finished.emit()



class NrrdWriterWorker(QObject):  
    finished = pyqtSignal()
    path = None
    array = None 
    header = None 
   
    def run(self): 
        nrrd.write(self.path, self.array, self.header)
        self.finished.emit()
        


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setOrganizationName("VisGroup Uni Jena")
    app.setOrganizationDomain("vis.uni-jena.de")
    app.setApplicationName("CarotidAnalyzer")
    app.setStyle("Fusion")
    win = CarotidAnalyzer()
    win.show()
    sys.exit(app.exec())