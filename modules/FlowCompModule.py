import os
import vtk
import numpy as np
import pyqtgraph as pg

from glob import glob
from PyQt5.QtWidgets import QWidget, QGridLayout, QHBoxLayout, QVBoxLayout, QLabel, QSplitter, QComboBox, QCheckBox, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5 import QtCore, QtGui
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from defaults import *

MAP_FIELD_NAMES = ['WSS_systolic', 'WSS_diastolic', 'longitudinal_WSS_systolic', 'longitudinal_WSS_diastolic']
COLORMAP_NAMES = ['Virids', 'Cividis', 'Plasma', 'Blues']
COLORMAP_KEYS = ['viridis', 'cividis', 'plasma', 'CET-L12']

# TODO move from global namespace
ctf_viridis = vtk.vtkColorTransferFunction()
ctf_viridis.SetColorSpaceToLab()
ctf_viridis.AddRGBPoint(1.0, 0.39, 0.2, 0.18)
ctf_viridis.AddRGBPoint(0.75, 0.51, 0.35, 0.47)
ctf_viridis.AddRGBPoint(0.5, 0.37, 0.59, 0.7)
ctf_viridis.AddRGBPoint(0.25, 0.31, 0.81, 0.67)
ctf_viridis.AddRGBPoint(0.0, 0.81, 0.95, 0.47)
def getLUTviridis(range_min, range_max):
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(255)
    lut.SetTableRange(range_min, range_max)
    for i in range(255):
        rgb = ctf_viridis.GetColor(i/255)
        lut.SetTableValue(i, rgb[0], rgb[1], rgb[2])
    return lut

class FlowCompModule(QWidget):
    """
    Visualization module for comparing a new geometry to similar geometries with computed flow field.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.working_dir = ""
        self.patient_data = []
        self.active_patient_dict = {'patient_ID':None}
        self.polling_counter = 0 # for cross-renderwindow camera synchronisation (every 10 frames)

        # latent space data
        self.map_datasets = []
        self.map_views = []
        self.comp_patient_containers = []
        self.surface_dataset_paths = [] # TODO integrate into container?
        self.flow_dataset_paths = []
        self.active_map_ids = []
        self.vessel_translations = {}
        self.vessel_rotations = {}
        self.vessel_translations_internal = {}
        self.vessel_rotations_internal = {}
        self.active_field_name = 'WSS_systolic'
        self.map_scale_min = {'WSS_systolic':0,   'WSS_diastolic':0,   'longitudinal_WSS_systolic':-50, 'longitudinal_WSS_diastolic':-50}
        self.map_scale_max = {'WSS_systolic':300, 'WSS_diastolic':300, 'longitudinal_WSS_systolic':0,   'longitudinal_WSS_diastolic':0}
        self.nr_latent_space_items = INITIAL_NR_MAPS

        # -------------------------------------
        # toolbar layout
        # -------------------------------------
        # set left / right side
        self.side_combobox = QComboBox()
        self.side_combobox.addItem("Left")
        self.side_combobox.addItem("Right")
        self.side_combobox.currentTextChanged[str].connect(lambda: self.loadPatient(self.active_patient_dict))
        
        # set active scalar field
        self.scalar_field_combobox = QComboBox()
        self.scalar_field_combobox.addItems(MAP_FIELD_NAMES)
        self.scalar_field_combobox.currentIndexChanged[int].connect(self.setScalarField)
        
        # set active colormap + range
        self.cmap = pg.colormap.get('viridis')
        self.colormap_combobox = QComboBox()
        self.colormap_combobox.addItems(COLORMAP_NAMES)
        self.colormap_combobox.currentIndexChanged[int].connect(self.setColorMap)
        self.color_bar = pg.ColorBarItem(
            colorMap=self.cmap,
            values=(self.map_scale_min[self.active_field_name], self.map_scale_max[self.active_field_name]),
            width=15,
            interactive=False,
            rounding=1,
            orientation='horizontal'
        )
        graphics_view = pg.GraphicsView()
        graphics_view.setBackground(None)
        graphics_view.setCentralWidget(self.color_bar)
        graphics_view.setMinimumSize(100, 36)
        graphics_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        
        # TODO set view flow
        # TODO set timestep
        # TODO set global plaque visibility
        self.link_cam_checkbox = QCheckBox("Link cameras")
        self.link_cam_checkbox.stateChanged[int].connect(self.linkCameras)
        
        # add all to toolbar
        self.toolbar_layout = QHBoxLayout()
        self.toolbar_layout.addWidget(QLabel("Side:"))
        self.toolbar_layout.addWidget(self.side_combobox)
        self.toolbar_layout.addWidget(VerticalLine())
        self.toolbar_layout.addWidget(QLabel("Colormap:"))
        self.toolbar_layout.addWidget(self.colormap_combobox)
        self.toolbar_layout.addWidget(QLabel("Active scalar field:"))
        self.toolbar_layout.addWidget(self.scalar_field_combobox)
        self.toolbar_layout.addWidget(graphics_view)
        self.toolbar_layout.addStretch(1)
        self.toolbar_layout.addWidget(self.link_cam_checkbox)

        # -------------------------------------
        # 3D views
        # -------------------------------------
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
        self.active_patient_camera.AddObserver("ModifiedEvent", self.linkedCameraModified)
        self.active_patient_view.GetRenderWindow().AddRenderer(self.active_patient_renderer)

        # active patient vtk pipelines
        self.active_patient_reader_lumen = vtk.vtkSTLReader()
        self.active_patient_reader_plaque = vtk.vtkSTLReader()

        self.active_patient_transform = vtk.vtkTransform()
        self.transform_filter_lumen = vtk.vtkTransformFilter()
        self.transform_filter_lumen.SetInputConnection(self.active_patient_reader_lumen.GetOutputPort())
        self.transform_filter_lumen.SetTransform(self.active_patient_transform)
        self.transform_filter_plaque = vtk.vtkTransformFilter()
        self.transform_filter_plaque.SetInputConnection(self.active_patient_reader_plaque.GetOutputPort())
        self.transform_filter_plaque.SetTransform(self.active_patient_transform)

        self.active_patient_mapper_lumen = vtk.vtkPolyDataMapper()
        self.active_patient_mapper_plaque = vtk.vtkPolyDataMapper()
        self.active_patient_mapper_lumen.SetInputConnection(self.transform_filter_lumen.GetOutputPort())
        self.active_patient_mapper_plaque.SetInputConnection(self.transform_filter_plaque.GetOutputPort())

        self.active_patient_actor_lumen = vtk.vtkActor()
        self.active_patient_actor_plaque = vtk.vtkActor()
        self.active_patient_actor_lumen.SetMapper(self.active_patient_mapper_lumen)
        self.active_patient_actor_plaque.SetMapper(self.active_patient_mapper_plaque)
        self.active_patient_actor_lumen.GetProperty().SetColor(COLOR_LUMEN)
        self.active_patient_actor_plaque.GetProperty().SetColor(COLOR_PLAQUE)

        # comparison patients vtk objects
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.comp_patient_view = QVTKRenderWindowInteractor()
        self.comp_patient_view.SetInteractorStyle(style)
        self.background_renderer = vtk.vtkRenderer()
        self.background_renderer.SetBackground(1,1,1)
        self.comp_patient_view.GetRenderWindow().AddRenderer(self.background_renderer)

        # 3D views vertical splitter
        self.splitter_3D = QSplitter(self)
        self.splitter_3D.addWidget(self.active_patient_view)
        self.splitter_3D.addWidget(self.comp_patient_view)

        # -------------------------------------
        # 2D view
        # -------------------------------------
        # latent space view, displays the maps
        self.latent_space_widget = ScrollableGraphicsLayoutWidget()
        self.latent_space_widget.scrolled_in.connect(self.decreaseMaps)
        self.latent_space_widget.scrolled_out.connect(self.increaseMaps)

        # 3D/latent horizontal splitter
        self.splitter_horizontal = QSplitter(self)
        self.splitter_horizontal.setOrientation(Qt.Vertical)
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

        # --------------------------------------
        # Load items that can be displayed
        # --------------------------------------
        # clear current view
        self.reset()

        # load latent space cases
        self.map_datasets.clear()
        self.surface_dataset_paths.clear()
        surface_file_pattern = os.path.join(self.working_dir, "flow_wall_data", "patient*_wss.vtu")
        for filename in glob(surface_file_pattern):
            map_file = filename[:-4] + "_map_images.npz"
            if os.path.exists(map_file):
                self.surface_dataset_paths.append(filename)
                self.map_datasets.append(np.load(map_file))

        # create all map widgets with the currently active image
        levels = (self.map_scale_min[self.active_field_name], self.map_scale_max[self.active_field_name])
        self.map_views = []
        for id, map_dataset in enumerate(self.map_datasets):
            # create view of map as an image item
            img_data = map_dataset[self.active_field_name]
            map_view = LatentSpaceItem(id, img_data, levels, self.cmap)
            map_view.clicked[int].connect(self.mapClicked)
            self.map_views.append(map_view)

        # initial latent space view, unsorted
        self.nr_latent_space_items = min(INITIAL_NR_MAPS, len(self.map_views))
        for i in range(self.nr_latent_space_items):
            self.latent_space_widget.addItem(self.map_views[i], row=0, col=i)

        # --------------------------------------
        # Load translations and rotations
        # --------------------------------------
        # flow database
        self.vessel_translations = {}
        self.vessel_rotations = {}
        registration_file = os.path.join(self.working_dir, "TransRot.txt")
        if os.path.exists(registration_file):
            with open(registration_file) as f:
                trans_rot = f.read().splitlines()
            for i in range(0, len(trans_rot), 3):
                identifier = trans_rot[i]
                translatation = trans_rot[i+1].split("t ")[1].split()
                rotation = trans_rot[i+2].split("r ")[1].split()
                self.vessel_translations[identifier] = [float(item) for item in translatation]
                self.vessel_rotations[identifier] = [float(item) for item in rotation]

        # new patients
        self.vessel_translations_internal = {}
        self.vessel_rotations_internal = {}
        registration_file = os.path.join(self.working_dir, "TransRot_internal.txt")
        if os.path.exists(registration_file):
            with open(registration_file) as f:
                trans_rot = f.read().splitlines()
            for i in range(0, len(trans_rot), 3):
                identifier = trans_rot[i]
                translatation = trans_rot[i+1].split("t ")[1].split()
                rotation = trans_rot[i+2].split("r ")[1].split()
                self.vessel_translations_internal[identifier] = [float(item) for item in translatation]
                self.vessel_rotations_internal[identifier] = [float(item) for item in rotation]


    def increaseMaps(self):
        if self.nr_latent_space_items < len(self.map_views)-1:
            map_view = self.map_views[self.nr_latent_space_items]
            self.latent_space_widget.addItem(map_view, row=0, col=self.nr_latent_space_items)
            self.nr_latent_space_items += 1
    

    def decreaseMaps(self):
        if self.nr_latent_space_items > 1:
            self.nr_latent_space_items -= 1
            map_view = self.map_views[self.nr_latent_space_items]
            self.latent_space_widget.removeItem(map_view)


    def mapClicked(self, id):
        img_box_clicked = self.map_views[id]
        id = img_box_clicked.id
        if id not in self.active_map_ids:
            # box is new -> activate
            self.active_map_ids.append(img_box_clicked.id)
            img_box_clicked.setActivated(True)

            # create a container for 3D vis of selected map
            case_identifier = os.path.basename(self.surface_dataset_paths[id])[:-8]
            try:
                translation = self.vessel_translations[case_identifier]
                rotation = self.vessel_rotations[case_identifier]
                # print("Loading surface and transforms for", case_identifier)
            except:
                print("Warning: No translation/rotation found for", case_identifier)
                translation = None
                rotation = None
            container = LatentSpace3DContainer(
                surface_file_path=self.surface_dataset_paths[id],
                active_field_name=self.active_field_name,
                scale_min=self.map_scale_min[self.active_field_name],
                scale_max=self.map_scale_max[self.active_field_name], # TODO provide current levels
                trans=translation,
                rot=rotation
                )
            if self.link_cam_checkbox.isChecked():
                container.useLinkedCamera(self.active_patient_camera)
            self.comp_patient_containers.append(container)
            self.comp_patient_view.GetRenderWindow().AddRenderer(container.renderer)
        
        else:
            # box is already active -> deactive
            index = self.active_map_ids.index(id)
            self.active_map_ids.pop(index)
            img_box_clicked.setActivated(False)

            # remove actor, camera, renderer
            container = self.comp_patient_containers.pop(index)
            self.comp_patient_view.GetRenderWindow().RemoveRenderer(container.renderer)

        # update the active items, update scenes
        nr_maps = len(self.active_map_ids)
        if nr_maps > 0:
            nr_rows = int(np.rint(np.sqrt(nr_maps)))
            nr_cols = int(np.ceil(nr_maps / nr_rows))
            h = 1.0 / nr_rows # single viewport height
            w = 1.0 / nr_cols # single viewport width
        for i in range(nr_maps):
            id = self.active_map_ids[i]
            self.map_views[id].setIdText(i+1)

            # update viewports
            row = nr_rows - 1 - int(i / nr_cols) # top -> bottom rows
            col = int(i % nr_cols)
            self.comp_patient_containers[i].renderer.SetViewport(col * w, row * h, (col + 1) * w, (row + 1) * h)
            self.comp_patient_containers[i].setIdText(i+1)

        # render all comparison views
        self.comp_patient_view.GetRenderWindow().Render()


    def loadPatient(self, patient_dict):
        self.active_patient_dict = patient_dict
        patient_id = self.active_patient_dict['patient_ID']
        if patient_id == None:
            return

        # load the models
        if self.side_combobox.currentText() == "Left":
            case_identifier = patient_id + "_left"
            lumen_path = self.active_patient_dict['lumen_model_left']
            plaque_path = self.active_patient_dict['plaque_model_left']
        else:
            case_identifier = patient_id + "_right"
            lumen_path = self.active_patient_dict['lumen_model_right']
            plaque_path = self.active_patient_dict['plaque_model_right']

        if lumen_path:
            self.active_patient_reader_lumen.SetFileName(lumen_path)
            self.active_patient_reader_lumen.Update()
            self.active_patient_renderer.AddActor(self.active_patient_actor_lumen)
        else:
            self.active_patient_renderer.RemoveActor(self.active_patient_actor_lumen)
        
        if plaque_path:
            self.active_patient_reader_plaque.SetFileName(plaque_path)
            self.active_patient_reader_plaque.Update()
            self.active_patient_renderer.AddActor(self.active_patient_actor_plaque)
        else:
            self.active_patient_renderer.RemoveActor(self.active_patient_actor_plaque)

        # load and apply transform
        rotation_mat = np.eye(4)
        translation_mat = np.eye(4)
        try:
            trans = self.vessel_translations_internal[case_identifier]
            rot = self.vessel_rotations_internal[case_identifier]
            translation_mat[0:3, 3] = np.array(trans)
            rotation_mat[0:3, 0:3] = np.reshape(rot, (3,3), order='F') # matrix is given column-wise
        except:
            print("Warning: No translation/rotation found for", case_identifier)
        transform_mat = rotation_mat @ translation_mat # translate, then rotate
        transform_mat = transform_mat.flatten(order='C') # row-wise input for vtk
        self.active_patient_transform.SetMatrix(transform_mat)

        # reset camera
        self.active_patient_camera.SetPosition(0, 0, -100)
        self.active_patient_camera.SetFocalPoint(0, 0, 0)
        self.active_patient_camera.SetViewUp(0, -1, 0)
        self.active_patient_renderer.ResetCamera()

        # TODO sort latent space

        self.active_patient_view.GetRenderWindow().Render()

    
    def setScalarField(self, index):
        self.active_field_name = MAP_FIELD_NAMES[index]
        levels = (self.map_scale_min[self.active_field_name], self.map_scale_max[self.active_field_name])

        # set field on 3D views
        for container in self.comp_patient_containers:
            container.setScalars(self.active_field_name)
            container.mapper.SetScalarRange(levels)
        self.comp_patient_view.GetRenderWindow().Render()
        
        # set field on map views
        for i in range(len(self.map_datasets)):
            img = self.map_views[i].img
            img.setImage(self.map_datasets[i][self.active_field_name])
            img.setLevels(levels)

    
    def setColorMap(self, index):
        self.cmap = pg.colormap.get(COLORMAP_KEYS[index])
        self.color_bar.setColorMap(self.cmap)
        for map_view in self.map_views:
            map_view.img.setColorMap(self.cmap)
        
    
    def linkCameras(self, link):
        if link:
            for container in self.comp_patient_containers:
                container.useLinkedCamera(self.active_patient_camera)
        else:
            for container in self.comp_patient_containers:
                container.useOwnCamera()
        self.comp_patient_view.GetRenderWindow().Render()

    
    def linkedCameraModified(self, obj, ev):
        self.polling_counter += 1
        if self.polling_counter > 10 and self.link_cam_checkbox.isChecked():
            self.polling_counter = 0
            self.comp_patient_view.GetRenderWindow().Render()
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
        # remove active patient
        self.active_patient_renderer.RemoveActor(self.active_patient_actor_lumen)
        self.active_patient_renderer.RemoveActor(self.active_patient_actor_plaque)

        # delete active 3D views of map
        for id in self.active_map_ids:
            self.map_views[id].setActivated(False)
        self.active_map_ids.clear()
        for c in self.comp_patient_containers:
            self.comp_patient_view.GetRenderWindow().RemoveRenderer(c.renderer)
        self.comp_patient_containers.clear()

        # remove map views from latent space navigator but keep map cache
        self.latent_space_widget.clear()

        # render views
        self.active_patient_view.GetRenderWindow().Render()
        self.comp_patient_view.GetRenderWindow().Render()


    def close(self):
        self.active_patient_view.Finalize()
        self.comp_patient_view.Finalize()



class ScrollableGraphicsLayoutWidget(pg.GraphicsLayoutWidget):
    """
    A pyqtgraph GraphicsLayoutWidget that emits wheel events.
    """
    scrolled_in = pyqtSignal()
    scrolled_out = pyqtSignal()
    def wheelEvent(self, ev):
        if ev.angleDelta().y() < 0:
            self.scrolled_out.emit()
            ev.accept()
        elif ev.angleDelta().y() > 0:
            self.scrolled_in.emit()
            ev.accept()
        else:
            ev.ignore() 



class LatentSpaceItem(pg.ViewBox):
    clicked = pyqtSignal(int)
    def __init__(self, id, img_data, levels, colormap, parent=None):
        super().__init__(parent=parent, border=None, lockAspect=True, enableMouse=False, enableMenu=False, defaultPadding=0)
        self.id = id

        # create internal items
        self.bar_stack = BarStackItem(img_data.shape[0], img_data.shape[1], [1.0, 1.0, 1.0], ['r', 'g', 'b'], thickness=50.0)
        self.img = pg.ImageItem(img_data)
        self.img.setLevels(levels)
        self.img.setColorMap(colormap)
        self.identifier_text_item = pg.TextItem(
            text='0', 
            color=(0, 0, 0), 
            anchor=(0.5, 1), 
            border=pg.mkPen(0, 0, 0), 
            fill=pg.mkBrush(255, 255, 255)
        )
        self.identifier_text_item.setPos(img_data.shape[1]*0.5, img_data.shape[0]*0.05)


        # add items to the view
        self.addItem(self.bar_stack)
        self.addItem(self.img)


    def mousePressEvent(self, ev):
        self.clicked.emit(self.id)

    
    def setActivated(self, activate):
        if activate:
            self.img.setBorder(pg.mkPen((100, 100, 100), width=3))
            self.addItem(self.identifier_text_item)
        else:
            self.img.setBorder(None)
            self.removeItem(self.identifier_text_item)

    
    def setIdText(self, nr):
            self.identifier_text_item.setText(str(nr))



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



class LatentSpace3DContainer():
    """
    Provides access to 3D items (camera, renderer, actor, mapper...)
    of one comparison case.
    """
    def __init__(self, surface_file_path, active_field_name, scale_min, scale_max, trans=None, rot=None):
        # id text
        self.identifier_text = vtk.vtkTextActor()
        p = self.identifier_text.GetTextProperty()
        p.SetColor(0, 0, 0)
        p.SetFontSize(20)
        p.SetBackgroundColor(1, 1, 1)

        # 3D surface object
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(surface_file_path)
        reader.Update()
        self.surface = reader.GetOutput()
        self.surface.GetPointData().SetActiveScalars(active_field_name)

        # transform
        rotation_mat = np.eye(4)
        rotation_mat[0:3, 0:3] = np.reshape(rot, (3,3), order='F') # matrix is given column-wise
        translation_mat = np.eye(4)
        translation_mat[0:3, 3] = np.array(trans)
        transform_mat = rotation_mat @ translation_mat # translate, then rotate
        transform_mat = transform_mat.flatten(order='C') # row-wise input for vtk

        transform = vtk.vtkTransform()
        transform.SetMatrix(transform_mat)
        transform_filter = vtk.vtkTransformFilter()
        transform_filter.SetInputData(self.surface)
        transform_filter.SetTransform(transform)

        # surface mapper/actor
        self.mapper = vtk.vtkDataSetMapper()
        self.mapper.SetInputConnection(transform_filter.GetOutputPort())
        self.mapper.SetScalarRange(scale_min, scale_max)
        self.mapper.SetLookupTable(getLUTviridis(0, 1)) # TODO use current LUT
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)

        # create a renderer, save own camera (viewports will be set later)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1, 1, 1)
        self.camera = self.renderer.GetActiveCamera()
        self.camera.SetPosition(0, 0, -100)
        self.camera.SetFocalPoint(0, 0, 0)
        self.camera.SetViewUp(0, -1, 0)
        self.renderer.AddActor(self.actor)
        self.renderer.AddActor(self.identifier_text)
        self.renderer.ResetCamera()
    
    def setIdText(self, identifier):
        self.identifier_text.SetInput(str(identifier))
    
    def setScalars(self, field_name):
        self.surface.GetPointData().SetActiveScalars(field_name)

    def setColormap(self, LUT):
        self.mapper.SetLookupTable(getLUTviridis(0, 1)) # TODO use current LUT

    def setColormapRange(self, scale_min, scale_max):
        self.mapper.SetScalarRange(scale_min, scale_max)

    def useLinkedCamera(self, cam):
        self.renderer.SetActiveCamera(cam)

    def useOwnCamera(self):
        # keep current view
        c = self.renderer.GetActiveCamera()
        self.camera.SetPosition(c.GetPosition())
        self.camera.SetFocalPoint(c.GetFocalPoint())
        self.camera.SetViewUp(c.GetViewUp())
        self.renderer.SetActiveCamera(self.camera)



from PyQt5.QtWidgets import QFrame
class VerticalLine(QFrame):
    """
    A vertical line for use in Qt Layouts.
    """
    def __init__(self):
        super(VerticalLine, self).__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)