import os
import vtk
import numpy as np
import pyqtgraph as pg

from glob import glob
from PyQt5.QtWidgets import QWidget, QGridLayout, QHBoxLayout, QVBoxLayout, QLabel, QSplitter, QComboBox
from PyQt5 import QtCore, QtGui
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from defaults import *

MAP_FIELD_NAMES = ['WSS_systolic', 'WSS_diastolic', 'longitudinal_WSS_systolic', 'longitudinal_WSS_diastolic']

class FlowCompModule(QWidget):
    """
    Visualization module for comparing a new geometry to similar geometries with computed flow field.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.working_dir = ""
        self.patient_data = []
        self.active_patient_dict = {'patient_ID':None}

        # latent space data
        self.map_datasets = []
        self.map_views = []
        self.active_field_name = 'WSS_systolic'
        self.map_scale_min = {'WSS_systolic':0,   'WSS_diastolic':0,   'longitudinal_WSS_systolic':-50, 'longitudinal_WSS_systolic':-50}
        self.map_scale_max = {'WSS_systolic':300, 'WSS_diastolic':300, 'longitudinal_WSS_systolic':0,   'longitudinal_WSS_systolic':0}
        self.nr_latent_space_items = 10 # initial number of displayed maps

        # toolbar layout
        self.side_combobox = QComboBox()
        self.side_combobox.addItem("Left")
        self.side_combobox.addItem("Right")
        self.side_combobox.currentTextChanged[str].connect(lambda: self.loadPatient(self.active_patient_dict))
        # TODO set active scalar field
        # TODO set active colormap + range
        # TODO set view flow
        # TODO set timestep
        # TODO set global plaque visibility
        self.toolbar_layout = QHBoxLayout()
        self.toolbar_layout.addWidget(self.side_combobox)

        # active patient model view
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.active_patient_view = QVTKRenderWindowInteractor()
        self.active_patient_view.SetInteractorStyle(style)
        self.active_patient_renderer = vtk.vtkRenderer()
        self.active_patient_renderer.SetBackground(1,1,1)
        self.active_patient_camera = self.active_patient_renderer.GetActiveCamera()
        self.active_patient_camera.SetPosition(0, 0, -100)
        self.active_patient_camera.SetFocalPoint(0, 0, 0)
        self.active_patient_camera.SetViewUp(0, -1, 0)
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

        # comparison patients model view
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.comp_patient_view = QVTKRenderWindowInteractor()
        self.comp_patient_view.SetInteractorStyle(style)
        self.comp_patient_renderer = vtk.vtkRenderer() # TODO make lists
        self.comp_patient_renderer.SetBackground(1,1,1)
        self.comp_patient_camera = self.comp_patient_renderer.GetActiveCamera()
        self.comp_patient_camera.SetPosition(0, 0, -100)
        self.comp_patient_camera.SetFocalPoint(0, 0, 0)
        self.comp_patient_camera.SetViewUp(0, -1, 0)
        self.comp_patient_view.GetRenderWindow().AddRenderer(self.comp_patient_renderer)

        # comparison patients vtk pipelines
        self.comp_patient_reader_lumen = vtk.vtkSTLReader() # TODO make lists, TODO unstructured grid readers
        self.comp_patient_reader_plaque = vtk.vtkSTLReader()
        self.comp_patient_mapper_lumen = vtk.vtkPolyDataMapper()
        self.comp_patient_mapper_plaque = vtk.vtkPolyDataMapper()
        self.comp_patient_mapper_lumen.SetInputConnection(self.comp_patient_reader_lumen.GetOutputPort())
        self.comp_patient_mapper_plaque.SetInputConnection(self.comp_patient_reader_plaque.GetOutputPort())
        self.comp_patient_actor_lumen = vtk.vtkActor()
        self.comp_patient_actor_plaque = vtk.vtkActor()
        self.comp_patient_actor_lumen.SetMapper(self.comp_patient_mapper_lumen)
        self.comp_patient_actor_plaque.SetMapper(self.comp_patient_mapper_plaque)
        self.comp_patient_actor_lumen.GetProperty().SetColor(COLOR_LUMEN) # TODO use active colormapping
        self.comp_patient_actor_plaque.GetProperty().SetColor(COLOR_PLAQUE)

        # 3D views vertical splitter
        self.splitter_3D = QSplitter(self)
        self.splitter_3D.addWidget(self.active_patient_view)
        self.splitter_3D.addWidget(self.comp_patient_view)

        # latent space view
        # self.latent_space_widget = pg.GraphicsLayoutWidget()
        self.latent_space_layout = QHBoxLayout()
        self.latent_space_layout.setContentsMargins(0, 0, 0, 0)
        self.latent_space_layout.setSpacing(0)
        self.latent_space_widget = QWidget()
        self.latent_space_widget.setLayout(self.latent_space_layout)

        # 3D/latent horizontal splitter
        self.splitter_horizontal = QSplitter(self)
        self.splitter_horizontal.setOrientation(QtCore.Qt.Vertical)
        self.splitter_horizontal.addWidget(self.splitter_3D)
        self.splitter_horizontal.addWidget(self.latent_space_widget)
        self.splitter_horizontal.setStretchFactor(0, 3)
        self.splitter_horizontal.setStretchFactor(1, 1)

        # top layout
        self.top_layout = QVBoxLayout(self)
        self.top_layout.addLayout(self.toolbar_layout)
        self.top_layout.addWidget(self.splitter_horizontal)

        # start interactors
        self.active_patient_view.Initialize()
        self.active_patient_view.Start()
        self.comp_patient_view.Initialize()
        self.comp_patient_view.Start()


    def setWorkingDir(self, dir, patient_data):
        self.working_dir = dir
        self.patient_data = patient_data

        # load all map image stacks
        map_file_pattern = os.path.join(self.working_dir, "flow_wall_data", "patient*_map_images.npz")
        self.map_datasets = [np.load(f) for f in glob(map_file_pattern)]

        # create all map widgets with the currently active image
        levels = (self.map_scale_min[self.active_field_name], self.map_scale_max[self.active_field_name])
        self.map_views = []
        for map_dataset in self.map_datasets:
            # create image item
            img_data = map_dataset[self.active_field_name]
            img = pg.ImageItem(img_data)
            img.setLevels(levels)
            img.setColorMap(pg.colormap.get('viridis')) # TODO use active colormap

            # create bar stack item
            bar_stack = BarStackItem(img_data.shape[0], img_data.shape[1], [1.0, 1.0, 1.0], ['r', 'g', 'b'], thickness=50.0)

            # add to layout
            img_box = pg.ViewBox(border=None, lockAspect=True, enableMouse=False, enableMenu=False, defaultPadding=0)
            img_box.addItem(img)
            img_box.addItem(bar_stack)
            view = pg.GraphicsView()
            view.setCentralWidget(img_box)
            self.map_views.append(view)

        # reset layout
        while not self.latent_space_layout.isEmpty():
            self.latent_space_layout.removeWidget(self.latent_space_layout.itemAt(0))

        # initial latent space view, unsorted
        for i in range(self.nr_latent_space_items):
            self.latent_space_layout.addWidget(self.map_views[i])
    

    def clicked(self, ev):
        print("Clicked!")


    def loadPatient(self, patient_dict):
        self.active_patient_dict = patient_dict
        patient_id = self.active_patient_dict['patient_ID']
        if self.side_combobox.currentText() == "Left":
            lumen_path = self.active_patient_dict['lumen_model_left']
            plaque_path = self.active_patient_dict['plaque_model_left']
        else:
            lumen_path = self.active_patient_dict['lumen_model_right']
            plaque_path = self.active_patient_dict['plaque_model_right']

        if lumen_path != None:
            self.active_patient_reader_lumen.SetFileName(lumen_path)
            self.active_patient_reader_lumen.Update()
            self.active_patient_renderer.AddActor(self.active_patient_actor_lumen)
        else:
            self.active_patient_renderer.RemoveActor(self.active_patient_actor_lumen)
        
        if plaque_path != None:
            self.active_patient_reader_plaque.SetFileName(plaque_path)
            self.active_patient_reader_plaque.Update()
            self.active_patient_renderer.AddActor(self.active_patient_actor_plaque)
        else:
            self.active_patient_renderer.RemoveActor(self.active_patient_actor_plaque)

        # TODO sort latent space

        self.active_patient_view.GetRenderWindow().Render()
        

    def showEvent(self, event):
        self.active_patient_view.Enable()
        self.active_patient_view.EnableRenderOn()
        super(FlowCompModule, self).showEvent(event)


    def hideEvent(self, event):
        self.active_patient_view.Disable()
        self.active_patient_view.EnableRenderOff()
        super(FlowCompModule, self).hideEvent(event)


    def reset(self):
        # TODO necessary?
        print("Removing all actors, resetting views")


    def close(self):
        self.active_patient_view.Finalize()
        self.comp_patient_view.Finalize()



class BarStackItem(pg.GraphicsObject):
    def __init__(self, y_offset, x_width, normalized_values_list, colors_list, thickness):
        super().__init__()
        self.y_offset = y_offset
        self.ends = np.array(normalized_values_list)[::-1] * x_width
        self.thickness = thickness
        self.colors = colors_list
        self.colors.reverse() # colors are internally ordered bottom->top
        self.generatePicture()
    
    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        for i in range(len(self.ends)):
            p.setBrush(pg.mkBrush(self.colors[i]))
            p.drawRect(QtCore.QRectF(0.0, self.y_offset + self.thickness*(i+1), self.ends[i], self.thickness))
        p.end()
    
    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())