import vtk
from vmtk import vmtkscripts
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QPushButton

from defaults import *

class CenterlineModuleTab(QWidget):
    """
    Tab view of a right OR left side carotid for centerline computation.
    """
    data_modified = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.lumen_active = False
        self.centerlines_active = False

        self.button_compute = QPushButton("Compute New Centerlines")
        self.button_compute.clicked.connect(self.computeCenterlines)
        self.centerline_view = QVTKRenderWindowInteractor(self)
        self.centerline_view.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1,1,1)
        self.centerline_view.GetRenderWindow().AddRenderer(self.renderer)

        self.slice_view_layout = QVBoxLayout(self)
        self.slice_view_layout.addWidget(self.button_compute)
        self.slice_view_layout.addWidget(self.centerline_view)

        # lumen vtk pipeline
        self.reader_lumen = vtk.vtkSTLReader()
        self.mapper_lumen = vtk.vtkPolyDataMapper()
        self.mapper_lumen.SetInputConnection(self.reader_lumen.GetOutputPort())
        self.actor_lumen = vtk.vtkActor()
        self.actor_lumen.SetMapper(self.mapper_lumen)
        self.actor_lumen.GetProperty().SetColor(COLOR_LUMEN)
        self.actor_lumen.GetProperty().SetOpacity(0.3)

        # centerline vtk pipeline
        self.reader_centerline = vtk.vtkXMLPolyDataReader()
        self.mapper_centerline = vtk.vtkPolyDataMapper()
        self.mapper_centerline.SetInputConnection(self.reader_centerline.GetOutputPort())
        self.actor_centerline = vtk.vtkActor()
        self.actor_centerline.SetMapper(self.mapper_centerline)
        self.actor_centerline.GetProperty().SetColor(0,0,0)
        self.actor_centerline.GetProperty().SetLineWidth(3)
        self.actor_centerline.GetProperty().RenderLinesAsTubesOn()

        self.centerline_view.Initialize()
        self.centerline_view.Start()


    def computeCenterlines(self):
        # TODO threading + windowless computation?
        if not self.lumen_active:
            print("No lumen to compute centerlines from.")
            return
        centerlines_script = vmtkscripts.vmtkCenterlines()
        centerlines_script.Surface = self.reader_lumen.GetOutput()
        centerlines_script.Execute()
        self.mapper_centerline.SetInputData(centerlines_script.Centerlines)
        self.centerline_view.GetRenderWindow().Render()
        self.centerlines_active = True
        self.data_modified.emit()


    def showEvent(self, event):
        self.centerline_view.Enable()
        self.centerline_view.EnableRenderOn()
        super(CenterlineModuleTab, self).showEvent(event)


    def hideEvent(self, event):
        self.centerline_view.Disable()
        self.centerline_view.EnableRenderOff()
        super(CenterlineModuleTab, self).hideEvent(event)
    

    def loadModels(self, lumen_file, centerline_file):
        if lumen_file:
            self.reader_lumen.SetFileName(lumen_file)
            self.renderer.AddActor(self.actor_lumen)
            self.lumen_active = True
            if centerline_file:
                self.reader_centerline.SetFileName(centerline_file)
                self.mapper_centerline.SetInputConnection(self.reader_centerline.GetOutputPort())
                self.renderer.AddActor(self.actor_centerline)
                self.centerlines_active = True
            else:
                self.renderer.RemoveActor(self.actor_centerline)
                self.centerlines_active = False
            self.renderer.ResetCamera()
        else:
            self.lumen_active = False
            self.centerlines_active = False
            self.renderer.RemoveActor(self.actor_lumen)
            self.renderer.RemoveActor(self.actor_centerline)
        self.centerline_view.GetRenderWindow().Render()

    def close(self):
        self.centerline_view.Finalize()



class CenterlineModule(QTabWidget):
    """
    Module for creating centerlines on vessel trees.
    User selects start/endpoints.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.patient_dict = None

        self.centerline_module_left = CenterlineModuleTab()
        self.centerline_module_right = CenterlineModuleTab()

        self.addTab(self.centerline_module_left, "Left")
        self.addTab(self.centerline_module_right, "Right")


    def load_patient(self, patient_dict):
        self.patient_dict = patient_dict
        self.centerline_module_left.loadModels(
            patient_dict['lumen_model_left'], patient_dict['centerlines_left'])
        self.centerline_module_right.loadModels(
            patient_dict['lumen_model_right'], patient_dict['centerlines_right'])

    def save(self):
        print("Centerline module saving changes made...")

    
    def discard(self):
        print("Centerline module discarding changes made...")
        self.load_patient(self.patient_dict)


    def close(self):
        self.centerline_module_left.close()
        self.centerline_module_right.close()