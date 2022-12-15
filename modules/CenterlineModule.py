import os

import vtk
from vmtk.vtkvmtkComputationalGeometryPython import vtkvmtkPolyDataCenterlines
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtCore import pyqtSignal
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
        self.centerlines = None
        self.DelaunayTessellation = None
        self.VoronoiDiagram = None
        self.PoleIds = None
        self.SourceIds = []
        self.TargetIds = []

        self.button_compute = QPushButton("Compute New Centerlines")
        self.button_compute.clicked.connect(self.computeCenterlines)
        self.interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        self.centerline_view = QVTKRenderWindowInteractor(self)
        self.centerline_view.SetInteractorStyle(self.interactor_style)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1,1,1)
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(0, 0, -100)
        cam.SetFocalPoint(0, 0, 0)
        cam.SetViewUp(0, -1, 0)
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

        # picking objects and observers
        self.pick_source = True
        self.picker = vtk.vtkPointPicker()
        self.pickPointEvent = self.interactor_style.AddObserver("RightButtonPressEvent", self.pickCenterlineEndPoint)
        self.actor_source = vtk.vtkActor()
        self.actors_targets = []

        # other vtk props
        self.text_patient = vtk.vtkTextActor()
        self.text_patient.SetInput("No segmentation file found for this side.")
        self.text_patient.SetDisplayPosition(10, 10)
        self.text_patient.GetTextProperty().SetColor(0, 0, 0)
        self.text_patient.GetTextProperty().SetFontSize(20)
        self.renderer.AddActor(self.text_patient)

        self.centerline_view.Initialize()
        self.centerline_view.Start()


    def pickCenterlineEndPoint(self, obj, event):
        # pick selected position
        x_screen, y_screen = self.centerline_view.GetEventPosition()
        self.picker.Pick(x_screen, y_screen, 0, self.renderer)
        position = self.picker.GetPickPosition()
        pointId = self.picker.GetPointId()
        print("Position: ", position)
        print("Point ID:", pointId)
        
        # create sphere actor
        sphere = vtk.vtkSphereSource()
        sphere.SetThetaResolution(20)
        sphere.SetPhiResolution(20)
        sphere.SetRadius(1)
        sphere.SetCenter(position)
        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere.GetOutputPort())

        if self.pick_source:
            self.SourceIds.append(pointId)
            self.actor_source.SetMapper(sphere_mapper)
            self.actor_source.GetProperty().SetColor(0.2, 1, 0.2)
            self.renderer.AddActor(self.actor_source)
            self.pick_source = False
        else:
            self.TargetIds.append(pointId)
            actor_target = vtk.vtkActor()
            actor_target.SetMapper(sphere_mapper)
            actor_target.GetProperty().SetColor(1, 0.2, 0.2)
            self.renderer.AddActor(actor_target)
            self.actors_targets.append(actor_target)
        self.centerline_view.GetRenderWindow().Render()
        

    def computeCenterlines(self):
        # TODO threading?
        if not self.lumen_active:
            print("No lumen to compute centerlines from.")
            return
        elif len(self.SourceIds) < 1:
            print("No source point specified.")
            return
        elif len(self.SourceIds) > 1:
            print("More than one source point specified.")
            return
        elif len(self.TargetIds) < 1:
            print("No target points specified.")
            return

        # create idlists from seepoints
        inletSeedIds = vtk.vtkIdList()
        outletSeedIds = vtk.vtkIdList()
        for id in self.SourceIds:
            inletSeedIds.InsertNextId(id)
        if self.TargetIds:
            for id in self.TargetIds:
                outletSeedIds.InsertNextId(id)

        # create centerline filter
        centerlineFilter = vtkvmtkPolyDataCenterlines()
        centerlineFilter.SetInputData(self.reader_lumen.GetOutput())
        centerlineFilter.SetSourceSeedIds(inletSeedIds)
        centerlineFilter.SetTargetSeedIds(outletSeedIds)
        centerlineFilter.SetRadiusArrayName('MaximumInscribedSphereRadius')
        centerlineFilter.SetFlipNormals(False)
        centerlineFilter.SetAppendEndPointsToCenterlines(False)
        centerlineFilter.SetStopFastMarchingOnReachingTarget(False) # could be enabled for 1 target
        centerlineFilter.SetSimplifyVoronoi(False)
        if self.DelaunayTessellation != None:
            centerlineFilter.GenerateDelaunayTessellationOff()
            centerlineFilter.SetDelaunayTessellation(self.DelaunayTessellation)
        if (self.VoronoiDiagram is not None) and (self.PoleIds is not None):
            centerlineFilter.GenerateVoronoiDiagramOff()
            centerlineFilter.SetVoronoiDiagram(self.VoronoiDiagram)
            centerlineFilter.SetPoleIds(self.PoleIds)
    
        # execute centerline filter
        centerlineFilter.SetCenterlineResampling(False)
        centerlineFilter.SetResamplingStepLength(1.0)
        centerlineFilter.Update()

        # cache output
        self.centerlines = centerlineFilter.GetOutput()
        self.VoronoiDiagram = centerlineFilter.GetVoronoiDiagram()
        self.DelaunayTessellation = centerlineFilter.GetDelaunayTessellation()
        self.PoleIds = centerlineFilter.GetPoleIds()

        # show output and propagate
        self.mapper_centerline.SetInputData(self.centerlines)
        self.renderer.AddActor(self.actor_centerline)
        self.centerline_view.GetRenderWindow().Render()
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
        self.DelaunayTessellation = None
        self.VoronoiDiagram = None
        self.PoleIds = None
        self.SourceIds.clear()
        self.TargetIds.clear()
        self.pick_source = True
        self.renderer.RemoveActor(self.actor_source)
        for actor in self.actors_targets:
            self.renderer.RemoveActor(actor)
        self.actor_source = vtk.vtkActor()
        self.actors_targets.clear()

        if lumen_file:
            self.reader_lumen.SetFileName("") # forces a reload
            self.reader_lumen.SetFileName(lumen_file)
            self.reader_lumen.Update()
            self.renderer.AddActor(self.actor_lumen)
            self.lumen_active = True
            self.text_patient.SetInput(os.path.basename(lumen_file)[:-4])
            if centerline_file:
                self.reader_centerline.SetFileName("")
                self.reader_centerline.SetFileName(centerline_file)
                self.mapper_centerline.SetInputConnection(self.reader_centerline.GetOutputPort())
                self.renderer.AddActor(self.actor_centerline)
                self.centerlines = self.reader_centerline.GetOutput()
            else:
                self.renderer.RemoveActor(self.actor_centerline)
                self.centerlines = None
            self.renderer.ResetCamera()
        else:
            self.lumen_active = False
            self.centerlines = None
            self.renderer.RemoveActor(self.actor_lumen)
            self.renderer.RemoveActor(self.actor_centerline)
            self.text_patient.SetInput("No segmentation file found for this side.")
        self.centerline_view.GetRenderWindow().Render()


    def saveChanges(self, path):
        # catch if one side has something to save, other side not
        if self.centerlines == None:
            return
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(path)
        writer.SetInputData(self.centerlines)
        writer.Write()


    def close(self):
        self.centerline_view.Finalize()



class CenterlineModule(QTabWidget):
    """
    Module for creating centerlines on vessel trees.
    User selects start/endpoints.
    """
    new_centerlines = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.patient_dict = None

        self.centerline_module_left = CenterlineModuleTab()
        self.centerline_module_right = CenterlineModuleTab()

        self.centerline_module_left.data_modified.connect(self.dataModifiedLeft)
        self.centerline_module_right.data_modified.connect(self.dataModifiedRight)

        self.addTab(self.centerline_module_right, "Right")
        self.addTab(self.centerline_module_left, "Left")


    def loadPatient(self, patient_dict):
        self.patient_dict = patient_dict
        self.centerline_module_right.loadModels(
            patient_dict['lumen_model_right'], patient_dict['centerlines_right'])
        self.centerline_module_left.loadModels(
            patient_dict['lumen_model_left'], patient_dict['centerlines_left'])


    def save(self):
        patient_ID = self.patient_dict['patient_ID']
        base_path  = self.patient_dict['base_path']
        path_right = os.path.join(base_path, "models", patient_ID + "_right_lumen_centerlines.vtp")
        path_left  = os.path.join(base_path, "models", patient_ID + "_left_lumen_centerlines.vtp")
        self.centerline_module_right.saveChanges(path_right)
        self.centerline_module_left.saveChanges(path_left)
        self.setTabText(0, "Right")
        self.setTabText(1, "Left")
        self.new_centerlines.emit()


    def dataModifiedRight(self):
        self.setTabText(0, "Right " + SYM_UNSAVED_CHANGES)


    def dataModifiedLeft(self):
        self.setTabText(1, "Left " + SYM_UNSAVED_CHANGES)
    
    
    def discard(self):
        self.loadPatient(self.patient_dict)
        self.setTabText(0, "Right")
        self.setTabText(1, "Left")


    def close(self):
        self.centerline_module_right.close()
        self.centerline_module_left.close()