import os

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vmtk import vmtkscripts
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QTabWidget
from PyQt5.QtGui import QColor
import pyqtgraph as pg

from defaults import *

# Override pyqtgraph defaults
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOption('antialias', True)

class StenosisClassifierTab(QWidget):
    """
    Tab view of a right OR left side carotid for stenosis classification.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.centerlines = None # raw centerlines (vtkPolyData)
        self.c_radii_lists = [] # processed centerline radii
        self.c_pos_lists = []   # processed centerline positions
        self.c_arc_lists = []   # processed centerline arc length (cumulated)

        # model view
        self.model_view = QVTKRenderWindowInteractor(self)
        self.model_view.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1,1,1)
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(0, 0, -100)
        cam.SetFocalPoint(0, 0, 0)
        cam.SetViewUp(0, -1, 0)
        self.model_view.GetRenderWindow().AddRenderer(self.renderer)

        # graph view
        self.widget_lineplots = pg.GraphicsLayoutWidget()
        self.lineplots = []

        # combine all in a layout
        self.top_layout = QHBoxLayout(self)
        self.top_layout.addWidget(self.widget_lineplots)
        self.top_layout.addWidget(self.model_view)

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

        # other vtk props
        self.text_patient = vtk.vtkTextActor()
        self.text_patient.SetInput("No lumen or centerlines file found for this side.")
        self.text_patient.SetDisplayPosition(10, 10)
        self.text_patient.GetTextProperty().SetColor(0, 0, 0)
        self.text_patient.GetTextProperty().SetFontSize(20)
        self.renderer.AddActor(self.text_patient)

        # start interactors
        self.model_view.Initialize()
        self.model_view.Start()


    def showEvent(self, event):
        self.model_view.Enable()
        self.model_view.EnableRenderOn()
        super(StenosisClassifierTab, self).showEvent(event)


    def hideEvent(self, event):
        self.model_view.Disable()
        self.model_view.EnableRenderOff()
        super(StenosisClassifierTab, self).hideEvent(event)
    

    def loadModels(self, lumen_file, centerline_file):
        if lumen_file and centerline_file:
            # load lumen
            self.reader_lumen.SetFileName("") # forces a reload
            self.reader_lumen.SetFileName(lumen_file)
            self.reader_lumen.Update()
            self.renderer.AddActor(self.actor_lumen)
            self.text_patient.SetInput(os.path.basename(lumen_file)[:-4])

            # load centerline
            self.reader_centerline.SetFileName("")
            self.reader_centerline.SetFileName(centerline_file)
            self.reader_centerline.Update()
            self.mapper_centerline.SetInputConnection(self.reader_centerline.GetOutputPort())
            self.renderer.AddActor(self.actor_centerline)
            self.centerlines = self.reader_centerline.GetOutput()
            self.__preprocessCenterlines(self.centerlines)
            self.plot_radii()

        else:
            # clear all
            self.renderer.RemoveActor(self.actor_lumen)
            self.renderer.RemoveActor(self.actor_centerline)
            self.centerlines = None
            self.text_patient.SetInput("No lumen or centerlines file found for this side.")

        # reset scene and render
        self.renderer.ResetCamera()
        self.model_view.GetRenderWindow().Render()

    def __preprocessCenterlines(self, centerlines):
        branch_extractor = vmtkscripts.vmtkBranchExtractor()
        branch_extractor.Centerlines = centerlines
        branch_extractor.Execute()

        c = branch_extractor.Centerlines
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(c)
        writer.SetFileName("C:\\Users\\Pepe Eulzer\\Desktop\\test_data\\centerlines_test.vtp")
        writer.Update()
        
        # lists for each line in centerlines
        # lines are ordered source->outlet
        self.c_pos_lists = []   # 3xn numpy arrays with point positions
        self.c_arc_lists = []   # 1xn numpy arrays with arc length along centerline (accumulated)
        self.c_radii_lists = [] # 1xn numpy arrays with maximal inscribed sphere radius

        # iterate all (global) lines
        # each line is a vtkIdList containing point ids in the right order
        l = centerlines.GetLines()        
        l.InitTraversal()
        for i in range(l.GetNumberOfCells()):
            pointIds = vtk.vtkIdList()
            l.GetNextCell(pointIds)

            # retrieve position data
            points = vtk.vtkPoints()
            centerlines.GetPoints().GetPoints(pointIds, points)
            p = vtk_to_numpy(points.GetData())
            self.c_pos_lists.append(p)

            # calculate arc len
            arc = p - np.roll(p, 1, axis=0)
            arc = np.sqrt((arc*arc).sum(axis=1))
            arc[0] = 0
            arc = np.cumsum(arc)
            self.c_arc_lists.append(arc)

            # retrieve radius data
            radii_flat = vtk_to_numpy(centerlines.GetPointData().GetArray('MaximumInscribedSphereRadius'))
            r = np.zeros(pointIds.GetNumberOfIds())
            for i in range(pointIds.GetNumberOfIds()):
                r[i] = radii_flat[pointIds.GetId(i)]
            self.c_radii_lists.append(r)

    
    def plot_radii(self):
        self.widget_lineplots.clear()
        lineplot_list = []
        for i in range(len(self.c_radii_lists)):
            lineplot = self.widget_lineplots.addPlot()
            lineplot.setLabel('left', "Minimal Radius (mm)")
            lineplot.showGrid(x=False, y=True, alpha=0.2)
            lineplot_list.append(lineplot)

            # correlation color of this glyph in [0,255]
            # color = (glyph.corr_color * 255.0).astype(np.int16)
            # color = QColor(color[0], color[1], color[2], 255)
            color = QColor(0, 0, 0, 255)
            line = lineplot.plot(x=self.c_arc_lists[i], y=self.c_radii_lists[i], pen=color)
            
            self.widget_lineplots.nextRow()

        # link axes
        for i in range(1, len(lineplot_list)):
            lineplot_list[i].setXLink(lineplot_list[0])
            lineplot_list[i].setYLink(lineplot_list[0])

        # set label
        lineplot_list[-1].setLabel('bottom', "Branch Length (mm)")


    def close(self):
        self.model_view.Finalize()



class StenosisClassifier(QTabWidget):
    """
    Visualization module for analyzing stenoses in vessel trees.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.patient_dict = None

        self.classifier_module_left = StenosisClassifierTab()
        self.classfifier_module_right = StenosisClassifierTab()

        self.addTab(self.classfifier_module_right, "Right")
        self.addTab(self.classifier_module_left, "Left")


    def loadPatient(self, patient_dict):
        self.patient_dict = patient_dict
        self.classfifier_module_right.loadModels(
            patient_dict['lumen_model_right'], patient_dict['centerlines_right'])
        self.classifier_module_left.loadModels(
            patient_dict['lumen_model_left'], patient_dict['centerlines_left'])


    def close(self):
        self.classfifier_module_right.close()
        self.classifier_module_left.close()