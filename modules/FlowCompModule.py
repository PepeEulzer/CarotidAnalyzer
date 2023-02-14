import os
import vtk
import numpy as np
import pyqtgraph as pg

from PyQt5.QtWidgets import QWidget

class FlowCompModule(QWidget):
    """
    Visualization module for comparing a new geometry to similar geometries with computed flow field.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.working_dir = ""
        self.patient_data = []
        self.active_patient_dict = {'patient_ID':None}
        

    def setWorkingDir(self, dir, patient_data):
        self.working_dir = dir
        self.patient_data = patient_data
        print("Looking for flow data in working dir...")

    def loadPatient(self, patient_dict):
        self.active_patient_dict = patient_dict
        print("Loading", self.active_patient_dict['patient_ID'])

    def close(self):
        print("Closing FlowComp Module")
        # call self.vtk_view.Finalize() on all vtk views