import os
import vtk
import numpy as np
import pyqtgraph as pg

from PyQt5.QtWidgets import QWidget, QGridLayout, QHBoxLayout, QLabel
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from defaults import *

class FlowCompModule(QWidget):
    """
    Visualization module for comparing a new geometry to similar geometries with computed flow field.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.working_dir = ""
        self.patient_data = []
        self.active_patient_dict = {'patient_ID':None}

        # active patient model view
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.active_patient_view = QVTKRenderWindowInteractor(self)
        self.active_patient_view.SetInteractorStyle(style)
        self.active_patient_renderer = vtk.vtkRenderer()
        self.active_patient_renderer.SetBackground(1,1,1)
        self.active_patient_camera = self.active_patient_renderer.GetActiveCamera()
        self.active_patient_camera.SetPosition(0, 0, -100)
        self.active_patient_camera.SetFocalPoint(0, 0, 0)
        self.active_patient_camera.SetViewUp(0, -1, 0)
        self.active_patient_camera.AddObserver("ModifiedEvent", self.cameraModifiedEvent)
        self.active_patient_view.GetRenderWindow().AddRenderer(self.active_patient_renderer)

        # active patient vtk pipelines
        self.active_patient_reader_lumen = vtk.vtkSTLReader()
        self.active_patient_reader_plaque = vtk.vtkSTLReader()
        self.active_patient_mapper_lumen = vtk.vtkPolyDataMapper()
        self.active_patient_mapper_plaque = vtk.vtkPolyDataMapper()
        self.active_patient_mapper_lumen.SetInputConnection(self.active_patient_reader_lumen.GetOutputPort())
        self.active_patient_mapper_plaque.SetInputConnection(self.active_patient_reader_plaque.GetOutputPort())
        self.active_patient_actor_lumen = vtk.vtkActor()
        self.active_patient_actor_plaque = vtk.vtkActor()
        self.active_patient_actor_lumen.SetMapper(self.active_patient_mapper_lumen)
        self.active_patient_actor_plaque.SetMapper(self.active_patient_mapper_plaque)
        self.active_patient_actor_lumen.GetProperty().SetColor(COLOR_LUMEN)
        self.active_patient_actor_plaque.GetProperty().SetColor(COLOR_PLAQUE)

        # "toolbar" layout
        self.toolbar_layout = QHBoxLayout()
        self.toolbar_layout.addWidget(QLabel("Toolbar"))

        # top layout (row, column, rowSpan, columnSpan)
        self.top_layout = QGridLayout(self)
        self.top_layout.addLayout(self.toolbar_layout, 0, 0, 1, 2)
        self.top_layout.addWidget(self.active_patient_view, 1, 0, 1, 1)

        # start interactors
        self.active_patient_view.Initialize()
        self.active_patient_view.Start()


    def showEvent(self, event):
        self.active_patient_view.Enable()
        self.active_patient_view.EnableRenderOn()
        super(FlowCompModule, self).showEvent(event)


    def hideEvent(self, event):
        self.active_patient_view.Disable()
        self.active_patient_view.EnableRenderOff()
        super(FlowCompModule, self).hideEvent(event)


    def setWorkingDir(self, dir, patient_data):
        self.working_dir = dir
        self.patient_data = patient_data
        print("Looking for flow data in working dir...")


    def loadPatient(self, patient_dict):
        self.active_patient_dict = patient_dict
        print("Loading", self.active_patient_dict['patient_ID'])


    def cameraModifiedEvent(self, obj, ev):
        print("Camera modified")


    def reset(self):
        print("Removing all actors, resetting views")


    def close(self):
        self.active_patient_view.Finalize()