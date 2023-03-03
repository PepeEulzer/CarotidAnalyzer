import os
import random
import json

import vtk
import numpy as np
import pyqtgraph as pg

from glob import glob
from PyQt5.QtWidgets import (QWidget, QGridLayout, QHBoxLayout, QVBoxLayout, QLabel, QSplitter, 
    QComboBox, QCheckBox, QSizePolicy, QDoubleSpinBox, QPushButton)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPalette, QColor
from PyQt5 import QtCore, QtGui
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util.numpy_support import vtk_to_numpy
from scipy.interpolate import interp1d

from defaults import *

MAP_FIELD_NAMES = ['WSS_systolic',
                   'WSS_diastolic',
                   'longitudinal_WSS_systolic',
                   'longitudinal_WSS_diastolic',
                   'velocity_systolic',
                   'velocity_diastolic']

MAP_DISPLAY_NAMES = ['Systolic WSS',
                   'Diastolic WSS',
                   'Systolic Backflow',
                   'Diastolic Backflow',
                   'Systolic Velocity',
                   'Diastolic Velocity']

MAP_TITLE_NAMES = ['<b>Systolic Wall Shear Stress [Pa]</b>',
                   '<b>Diastolic Wall Shear Stress [Pa]</b>',
                   '<b>Systolic Backflow (reverse WSS) [Pa]</b>',
                   '<b>Diastolic Backflow (reverse WSS) [Pa]</b>',
                   '<b>Systolic Velocity [m/s]</b>',
                   '<b>Diastolic Velocity [m/s]</b>']

COLORMAP_NAMES = ['Viridis', 'Cividis', 'Plasma', 'Blues']
COLORMAP_KEYS =  ['viridis', 'cividis', 'plasma', 'CET-L12']

HTML_SHAPE_COLOR = "#b3b3b3"
HTML_DIAMETER_COLOR = "#808080"
HTML_STENOSIS_COLOR = "#4d4d4d"

LANDMARK_ALPHA = 20 * np.pi / 180 # angle (rad) for default ACI-ACE layout
LANDMARK_RANGE = 15 # distance (mm) from bifurcation point for automatic landmarks
LANDMARK_TARGETS = vtk.vtkPoints()
LANDMARK_TARGETS.InsertNextPoint([0, 0, -LANDMARK_RANGE]) # ACC
LANDMARK_TARGETS.InsertNextPoint([-LANDMARK_RANGE*np.sin(LANDMARK_ALPHA), 0, LANDMARK_RANGE*np.cos(LANDMARK_ALPHA)]) # ACI
LANDMARK_TARGETS.InsertNextPoint([LANDMARK_RANGE*np.sin(LANDMARK_ALPHA), 0, LANDMARK_RANGE*np.cos(LANDMARK_ALPHA)]) # ACE


def getVTKLookupTable(cmap, nPts=512):
    # returns a vtkLookupTable from any given pyqtgraph colormap
    pg_lut = cmap.getLookupTable(nPts=nPts, mode=pg.ColorMap.FLOAT)
    vtk_lut = vtk.vtkLookupTable()
    vtk_lut.SetNumberOfTableValues(nPts)
    vtk_lut.SetTableRange(0, 1) # scaled by mapper
    for i in range(nPts):
        vtk_lut.SetTableValue(i, pg_lut[i,0], pg_lut[i,1], pg_lut[i,2])
    return vtk_lut


def getMetaInformation(meta_path):
    stenosis_degree = 0.0
    if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta_dict = json.load(f)
                stenosis_degree = np.round(float(meta_dict['stenosis_degree']), 1)
            except:
                print("FlowComp module could not load " + meta_path)
    return stenosis_degree


def processCenterline(centerline_path):
    """
    Loads a centerline and extracts relevant information.
    Returns:
    * centerline radii of ACC/ACI branch (np.array)
    * index of the bifurcation point
    * ACC landmark coordinates (10mm before bifurcation)
    * ACI landmark coordinates (10mm after bifurcation)
    * ACE landmark coordinates (10mm after bifurcation)
    """
    radii_array = None
    bifurcation_index = 0
    landmark_ACC = (0, 0, 0)
    landmark_bifurcation = (0, 0, 0)
    landmark_ACI = (0, 0, 0)

    if not os.path.exists(centerline_path):
        return radii_array, bifurcation_index, landmark_ACC, landmark_bifurcation, landmark_ACI

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(centerline_path)
    reader.Update()
    centerlines = reader.GetOutput()
    radii_flat = centerlines.GetPointData().GetArray('MaximumInscribedSphereRadius')

    if centerlines.GetLines().GetNumberOfCells() < 2:
        return radii_array, bifurcation_index, landmark_ACC, landmark_bifurcation, landmark_ACI

    # --------------------------------
    # iterate the first line (ACC/ACI)
    l = centerlines.GetLines()
    l.InitTraversal()
    pointIds = vtk.vtkIdList()
    l.GetNextCell(pointIds)

    # retrieve position data
    points = vtk.vtkPoints()
    centerlines.GetPoints().GetPoints(pointIds, points)
    p = vtk_to_numpy(points.GetData())

    # calculate arc len (x-axis)
    arc = p - np.roll(p, 1, axis=0)
    arc = np.sqrt((arc*arc).sum(axis=1))
    arc[0] = 0
    arc = np.cumsum(arc)

    # retrieve radius data (y-axis)
    radii = vtk.vtkDoubleArray()
    radii.SetNumberOfTuples(pointIds.GetNumberOfIds())
    radii_flat.GetTuples(pointIds, radii)
    radii_array = vtk_to_numpy(radii)

    # resample equidistant radii
    resample = interp1d(arc, radii, assume_sorted=True)
    x_samples = np.arange(0, arc[-1], 0.1) # must use same stepwidth (e.g. 0.1mm) for all arrays!
    radii_array_resampled = resample(x_samples)

    # --------------------------------
    # iterate second line (ACC/ACE)
    pointIds = vtk.vtkIdList()
    l.GetNextCell(pointIds)
    points2 = vtk.vtkPoints()
    centerlines.GetPoints().GetPoints(pointIds, points2)
    p2 = vtk_to_numpy(points2.GetData())

    # calculate ACE arc len
    arc2 = p2 - np.roll(p2, 1, axis=0)
    arc2 = np.sqrt((arc2*arc2).sum(axis=1))
    arc2[0] = 0
    arc2 = np.cumsum(arc2)

    # find last overlap index
    len1 = p.shape[0]
    len2 = p2.shape[0]
    if len1 < len2:
        overlap_mask = np.not_equal(p, p2[:len1])
    else:
        overlap_mask = np.not_equal(p[:len2], p2)
    overlap_mask = np.all(overlap_mask, axis=1) # AND over tuples
    bifurcation_index = np.searchsorted(overlap_mask, True) # first position where lines diverge

    # --------------------------------
    # retrieve landmarks around bifurcation
    bifurcation_arclen = arc[bifurcation_index]
    landmark_ACC = p[np.argmax(arc > bifurcation_arclen - LANDMARK_RANGE)]
    landmark_ACI = p[np.argmax(arc > bifurcation_arclen + LANDMARK_RANGE)]
    landmark_ACE = p2[np.argmax(arc2 > bifurcation_arclen + LANDMARK_RANGE)]

    # compute the correct bifurcation index for the resampled radii array
    bifurcation_index_resampled = np.searchsorted(x_samples >= bifurcation_arclen, True)
    return radii_array_resampled, bifurcation_index_resampled, landmark_ACC, landmark_ACI, landmark_ACE



class LatentSpaceDataSet(object):
    """
    A wrapper of all items related to a single latent space comparison case.
    Contains the map data set, paths to the 3D surfaces, streamlines, etc.
    Also computes and contains the similarity score given a new carotid model.
    Sorting these objects automatically sorts them by their similarity score.
    """
    # 2D map objects
    map_dataset = None  # np dict of all map images (wss_map_images.npz)
    map_view = None     # MapViewBox (pg.ViewBox) of the active map
    is_clicked = False  # indicates if the map is clicked (3D view active)

    # 3D objects (paths)
    surface_dataset_path = None      # path to surface file in flow folder, if exists (wss.vtu)
    systolic_streamline_path = None  # path to systolic streamlines, if exists (systolic.vtp)
    diastolic_streamline_path = None # path to diastolic streamlines, if exists (diastolic.vtp)

    # landmarks for registration (automatically extracted)
    landmark_ACC = None
    landmark_ACI = None
    landmark_ACE = None

    # similarity variables
    diameter_profile = None   # np.array of ACC/ACI diameters, equidistantly spaced
    bifurcation_index = 0     # index into diameter profile for registering at bifurcation
    stenosis_degree = 0.0     # highest stenosis degree of ACC/ACI branch in [0,100]
    similarity_shape = 0.0    # how similar is the latent space shape? [0,1]
    similarity_diameter = 0.0 # how similar is the diameter profile? [0,1]
    similarity_stenosis = 0.0 # how similar is the stenosis degree? [0,1]
    similarity_score = 0.0    # combined similarity score given weights for each component, value in [0,1]

    def __init__(self, case_id, centerline_path, stenosis_degree):
        self.case_identifier = case_id
        self.diameter_profile, self.bifurcation_index, \
            self.landmark_ACC, self.landmark_ACI, self.landmark_ACE = processCenterline(centerline_path)
        self.stenosis_degree = stenosis_degree

    def __eq__(self, other):
        return self.similarity_score == other.similarity_score

    def __lt__(self, other):
        return self.similarity_score < other.similarity_score

    def compareWithCase(self, distance_to_case, case_stenosis_degree, case_diameter_profile, case_bifurcation_index):
        if self.diameter_profile is None or case_diameter_profile is None:
            mean_diameter_distance = 1.5 # set to max
        else:
            # compare diameter profiles, register at bifurcation -> crop start/end to smaller array
            r1 = self.diameter_profile # internal radii (will be cropped)
            r2 = case_diameter_profile # external radii (will be cropped)
            i1 = self.bifurcation_index
            i2 = case_bifurcation_index
            if i1 < i2:
                r2 = r2[i2-i1:]
            else:
                r1 = r1[i1-i2:]
            l1 = r1.shape[0] # length of internal array
            l2 = r2.shape[0] # length of case array
            if l1 < l2:
                r2 = r2[:l1]
            else:
                r1 = r1[:l2]
            assert(r1.shape[0] == r2.shape[0])
            mean_diameter_distance = np.mean(np.abs(r1-r2))

        # each value is 1 - difference / max_difference
        self.similarity_shape = 1.0 - distance_to_case
        self.similarity_diameter = 1.0 - mean_diameter_distance / 1.5
        self.similarity_stenosis = 1.0 - abs(self.stenosis_degree - case_stenosis_degree) / 100.0
        if self.map_view is not None:
            self.map_view.updateScoreBars([self.similarity_shape, self.similarity_diameter, self.similarity_stenosis])

    def updateSimilarityScore(self, shape_weight=1, diam_weight=1, stenosis_weight=1):
        sum_weight = shape_weight + diam_weight + stenosis_weight
        if sum_weight == 0:
            self.similarity_score = 0
            return
        shape_weight /= sum_weight
        diam_weight /= sum_weight
        stenosis_weight /= sum_weight
        self.similarity_score = shape_weight * self.similarity_shape \
                                + diam_weight * self.similarity_diameter \
                                + stenosis_weight * self.similarity_stenosis



class FlowCompModule(QWidget):
    """
    Visualization module for comparing a new geometry to similar geometries with computed flow field.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # active state variables
        self.working_dir = ""
        self.patient_data = []
        self.active_patient_dict = {'patient_ID':None}
        self.polling_counter = 0 # for cross-renderwindow camera synchronisation (every 10 frames)

        # latent space data
        self.latent_space_datasets = []
        self.active_map_ids = []          # indices into above list, indicate active (clicked) cases
        self.comp_patient_containers = [] # LatentSpace3DContainer of all active 3D views
        self.nr_map_rows = 0 # will be filled later
        self.nr_map_cols = 0
        self.nr_latent_space_items = INITIAL_NR_MAPS

        # translation and rotation dicts for cases, key is the patient id (+ left/right)
        self.vessel_translations = {}
        self.vessel_rotations = {}
        self.vessel_translations_internal = {}
        self.vessel_rotations_internal = {}

        # case-case position differences after registration
        self.dist_mat = np.array([])   # symmetric distance matrix
        self.dist_mat_identifiers = [] # row/column names (case identifiers)

        # possible scalar fields, will be mapped to surface or streamlines with active colormap
        self.active_field_name = MAP_FIELD_NAMES[0]
        self.map_scale_min = {'WSS_systolic':0,
                              'WSS_diastolic':0,   
                              'longitudinal_WSS_systolic':-50,
                              'longitudinal_WSS_diastolic':-50,
                              'velocity_systolic':0,
                              'velocity_diastolic':0}
        self.map_scale_max = {'WSS_systolic':300,
                              'WSS_diastolic':300,
                              'longitudinal_WSS_systolic':0,
                              'longitudinal_WSS_diastolic':0,
                              'velocity_systolic':3,
                              'velocity_diastolic':3}

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
        self.scalar_field_combobox.addItems(MAP_DISPLAY_NAMES)
        self.scalar_field_combobox.currentIndexChanged[int].connect(self.setScalarField)
        
        # set active colormap + range
        self.cmap = pg.colormap.get('viridis')
        self.vtk_lut = getVTKLookupTable(self.cmap)
        self.colormap_combobox = QComboBox()
        self.colormap_combobox.addItems(COLORMAP_NAMES)
        self.colormap_combobox.currentIndexChanged[int].connect(self.setColorMap)
        levels = self.getLevels()
        self.levels_min_spinbox = QDoubleSpinBox()
        self.levels_min_spinbox.setSingleStep(10)
        self.levels_min_spinbox.setDecimals(1)
        self.levels_min_spinbox.setSuffix(" [Pa]")
        self.levels_min_spinbox.setRange(-999, levels[1])
        self.levels_min_spinbox.setValue(levels[0])
        self.levels_min_spinbox.valueChanged[float].connect(self.setLevelsMin)
        self.color_bar = pg.ColorBarItem(
            colorMap=self.cmap,
            values=levels,
            width=15,
            interactive=False,
            rounding=1,
            orientation='horizontal'
        )
        graphics_view = pg.GraphicsView()
        graphics_view.setBackground(None)
        graphics_view.setCentralWidget(self.color_bar)
        graphics_view.setMinimumSize(100, 36)
        graphics_view.setMaximumSize(1000, 36)
        self.levels_max_spinbox = QDoubleSpinBox()
        self.levels_max_spinbox.setSingleStep(10)
        self.levels_max_spinbox.setDecimals(1)
        self.levels_max_spinbox.setSuffix(" [Pa]")
        self.levels_max_spinbox.setRange(levels[0], 999)
        self.levels_max_spinbox.setValue(levels[1])
        self.levels_max_spinbox.valueChanged[float].connect(self.setLevelsMax)
        self.colormap_title = QLabel(MAP_TITLE_NAMES[0])

        # link/unlink cameras
        self.link_cam_checkbox = QCheckBox("Link cameras")
        self.link_cam_checkbox.stateChanged[int].connect(self.linkCameras)

        # reset selection
        self.reset_button = QPushButton("Reset selection")
        self.reset_button.clicked.connect(self.resetCompViews)
        
        # add all to toolbar (row, column, rowspan, columnspan)
        self.toolbar_layout = QGridLayout()
        self.toolbar_layout.addWidget(QLabel("Side:"), 0, 0, Qt.AlignBottom)
        self.toolbar_layout.addWidget(self.side_combobox, 1, 0)
        self.toolbar_layout.addWidget(VerticalLine(), 0, 1, 2, 1)
        self.toolbar_layout.addWidget(QLabel("Colormap:"), 0, 2, Qt.AlignBottom)
        self.toolbar_layout.addWidget(self.colormap_combobox, 1, 2)
        self.toolbar_layout.addWidget(QLabel("Active scalar field:"), 0, 3, Qt.AlignBottom)
        self.toolbar_layout.addWidget(self.scalar_field_combobox, 1, 3)
        self.toolbar_layout.addWidget(self.colormap_title, 0, 4, 1, 3, Qt.AlignHCenter | Qt.AlignBottom)
        self.toolbar_layout.addWidget(self.levels_min_spinbox, 1, 4)
        self.toolbar_layout.addWidget(graphics_view, 1, 5)
        self.toolbar_layout.addWidget(self.levels_max_spinbox, 1, 6)
        self.toolbar_layout.addWidget(VerticalLine(), 0, 7, 2, 1)
        self.toolbar_layout.setColumnStretch(8, 1)
        self.toolbar_layout.addWidget(self.reset_button, 0, 8, Qt.AlignRight)
        self.toolbar_layout.addWidget(self.link_cam_checkbox, 1, 8, Qt.AlignRight)

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
        self.active_patient_text = vtk.vtkTextActor()
        p = self.active_patient_text.GetTextProperty()
        p.SetColor(0, 0, 0)
        p.SetFontSize(20)
        p.SetBackgroundColor(1, 1, 1)

        self.active_patient_reader_lumen = vtk.vtkSTLReader()
        self.active_patient_reader_plaque = vtk.vtkSTLReader()

        normals_lumen = vtk.vtkPolyDataNormals()
        normals_lumen.ComputePointNormalsOn()
        normals_lumen.ComputeCellNormalsOff()
        normals_lumen.ConsistencyOn()
        normals_lumen.SplittingOff()
        normals_lumen.SetInputConnection(self.active_patient_reader_lumen.GetOutputPort())
        normals_plaque = vtk.vtkPolyDataNormals()
        normals_plaque.ComputePointNormalsOn()
        normals_plaque.ComputeCellNormalsOff()
        normals_plaque.ConsistencyOn()
        normals_plaque.SplittingOff()
        normals_plaque.SetInputConnection(self.active_patient_reader_plaque.GetOutputPort())

        self.transform_filter_lumen = vtk.vtkTransformFilter()
        self.transform_filter_lumen.SetInputConnection(normals_lumen.GetOutputPort())
        self.transform_filter_plaque = vtk.vtkTransformFilter()
        self.transform_filter_plaque.SetInputConnection(normals_plaque.GetOutputPort())

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
        self.active_patient_actor_plaque.GetProperty().SetOpacity(0.5)
        self.active_patient_actor_plaque.GetProperty().BackfaceCullingOn()

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
        # latent space legend, left to the maps
        self.latent_space_legend = QVBoxLayout()
        self.latent_space_legend.addStretch(1)
        self.latent_space_legend.addWidget(QLabel('<b>Comparison Metrics</b>'), alignment=Qt.AlignCenter)

        self.shape_checkbox = CheckBoxDecorator("Shape Similarity", HTML_SHAPE_COLOR)
        self.shape_checkbox.checkbox.clicked.connect(self.sortLatentSpace)
        self.latent_space_legend.addWidget(self.shape_checkbox, alignment=Qt.AlignLeft)

        self.diameter_checkbox = CheckBoxDecorator("Diameter Similarity", HTML_DIAMETER_COLOR)
        self.diameter_checkbox.checkbox.clicked.connect(self.sortLatentSpace)
        self.latent_space_legend.addWidget(self.diameter_checkbox, alignment=Qt.AlignLeft)

        self.stenosis_checkbox = CheckBoxDecorator("Stenosis Similarity", HTML_STENOSIS_COLOR)
        self.stenosis_checkbox.checkbox.clicked.connect(self.sortLatentSpace)
        self.latent_space_legend.addWidget(self.stenosis_checkbox, alignment=Qt.AlignLeft)
        self.latent_space_legend.addStretch(1)

        # latent space view, displays the maps
        self.latent_space_widget = ScrollableGraphicsLayoutWidget()
        self.latent_space_widget.sigDeviceTransformChanged.connect(lambda: self.fillMapView(0, False))
        self.latent_space_widget.scrolled_in.connect(self.decreaseMaps)
        self.latent_space_widget.scrolled_out.connect(self.increaseMaps)

        # combine legend and latent space map view
        latent_space_layout = QHBoxLayout()
        latent_space_layout.addLayout(self.latent_space_legend)
        latent_space_layout.addWidget(self.latent_space_widget)
        latent_space_layout.setContentsMargins(0, 0, 0, 0)
        latent_space_wrapper = QWidget()
        latent_space_wrapper.setLayout(latent_space_layout)

        # 3D/latent horizontal splitter
        self.splitter_horizontal = QSplitter(self)
        self.splitter_horizontal.setOrientation(Qt.Vertical)
        self.splitter_horizontal.addWidget(self.splitter_3D)
        self.splitter_horizontal.addWidget(latent_space_wrapper)
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
        self.patient_data = patient_data # contains all dicts on all patient file paths

        # --------------------------------------
        # Load items that can be displayed
        # --------------------------------------
        # clear current view
        self.resetAllViews()

        # load latent space cases
        self.latent_space_datasets.clear()
        surface_file_pattern = os.path.join(self.working_dir, "flow_wall_data", "patient*_wss.vtu")
        for surface_file in glob(surface_file_pattern):
            # if map and surface file exist -> create a case
            patient_id_lr = os.path.basename(surface_file)[:-8]
            patient_id = patient_id_lr.split('_left')[0].split('_right')[0]
            meta_path = os.path.join(self.working_dir, patient_id, "models", patient_id_lr + "_meta.txt")
            cent_path = surface_file.split("wss.vtu")[0] + "lumen_centerlines.vtp"
            stenosis_degree = getMetaInformation(meta_path)
            latent_space_dataset = LatentSpaceDataSet(patient_id_lr, cent_path, stenosis_degree)

            map_file = surface_file[:-4] + "_map_images.npz"
            if os.path.exists(map_file):
                latent_space_dataset.surface_dataset_path = surface_file
                latent_space_dataset.map_dataset = np.load(map_file)
            else:
                continue
            
            # load systolic streamlines if they exist
            systolic_streamline_file = surface_file[:-7] + "velocity_systolic.vtp"
            if os.path.exists(systolic_streamline_file):
                latent_space_dataset.systolic_streamline_path = systolic_streamline_file

            # load diastolic streamlines if they exist
            diastolic_streamline_file = surface_file[:-7] + "velocity_diastolic.vtp"
            if os.path.exists(diastolic_streamline_file):
                latent_space_dataset.diastolic_streamline_path = diastolic_streamline_file

            self.latent_space_datasets.append(latent_space_dataset)

        # create all map widgets with the currently active image
        for index, dataset in enumerate(self.latent_space_datasets):
            # create view of map as an image item
            img_data = dataset.map_dataset[self.active_field_name]
            map_view = MapViewBox(index, img_data, self.getLevels(), self.cmap)
            map_view.clicked[int].connect(self.mapClicked)
            dataset.map_view = map_view

        # initial latent space view, unsorted, will be filled automatically
        self.nr_latent_space_items = min(INITIAL_NR_MAPS, len(self.latent_space_datasets))

        # --------------------------------------
        # Load translations and rotations
        # --------------------------------------
        # load distance matrix
        matrix_file = os.path.join(self.working_dir, "distMatrix.txt")
        if os.path.exists(matrix_file):
            self.dist_mat = np.loadtxt(matrix_file, dtype=np.float32)
            self.dist_mat /= np.max(self.dist_mat) # normalize
        else:
            print("WARNING: Distance matrix could not be found.")
        self.dist_mat_identifiers = [] # list of case identifiers, rows/colums of matrix

        # flow database
        self.vessel_translations = {}
        self.vessel_rotations = {}
        self.vessel_distances = {}
        registration_file = os.path.join(self.working_dir, "TransRot.txt")
        if os.path.exists(registration_file):
            with open(registration_file) as f:
                trans_rot = f.read().splitlines()
            for i in range(0, len(trans_rot), 3):
                identifier = trans_rot[i]
                translation = trans_rot[i+1].split("t ")[1].split()
                rotation = trans_rot[i+2].split("r ")[1].split()
                self.vessel_translations[identifier] = [float(item) for item in translation]
                self.vessel_rotations[identifier] = [float(item) for item in rotation]
                self.dist_mat_identifiers.append(identifier)

        if len(self.dist_mat_identifiers) != self.dist_mat.shape[0]:
            print("WARNING: Distance matrix shape", self.dist_mat.shape[0],
                  "does not match number of cases", len(self.dist_mat_identifiers))

        # new patients
        self.vessel_translations_internal = {}
        self.vessel_rotations_internal = {}
        registration_file = os.path.join(self.working_dir, "TransRot_internal.txt")
        if os.path.exists(registration_file):
            with open(registration_file) as f:
                trans_rot = f.read().splitlines()
            for i in range(0, len(trans_rot), 3):
                identifier = trans_rot[i]
                translation = trans_rot[i+1].split("t ")[1].split()
                rotation = trans_rot[i+2].split("r ")[1].split()
                self.vessel_translations_internal[identifier] = [float(item) for item in translation]
                self.vessel_rotations_internal[identifier] = [float(item) for item in rotation]

    
    def fillMapView(self, nr_items_change=0, force_reload=False):
        self.nr_latent_space_items += nr_items_change
        container_w = self.latent_space_widget.size().width()
        container_h = self.latent_space_widget.size().height()
        nr_rows = 1
        nr_cols = self.nr_latent_space_items
        element_w = container_w / nr_cols
        element_h = 2 * element_w
        while element_h * 2 <= container_h:
            nr_rows += 1
            nr_cols = np.ceil(self.nr_latent_space_items / nr_rows)
            element_w = container_w / nr_cols
            element_h = 2 * element_w

        # clear the container and apply the layout (only if something changed)
        if force_reload or self.nr_map_rows != nr_rows or self.nr_map_cols != nr_cols:
            self.nr_map_rows = nr_rows
            self.nr_map_cols = nr_cols
            self.latent_space_widget.clear()
            for i in range(self.nr_latent_space_items):
                row = int(i / nr_cols)
                col = int(i % nr_cols)
                self.latent_space_widget.addItem(self.latent_space_datasets[i].map_view, row=row, col=col)


    def increaseMaps(self):
        if self.nr_latent_space_items < len(self.latent_space_datasets)-1:
            self.fillMapView(nr_items_change=1)
    

    def decreaseMaps(self):
        if self.nr_latent_space_items > 1:
            self.fillMapView(nr_items_change=-1)


    def mapClicked(self, index):
        ls_dataset = self.latent_space_datasets[index]
        if index not in self.active_map_ids:
            # box is new -> activate
            self.active_map_ids.append(index)
            ls_dataset.is_clicked = True
            ls_dataset.map_view.setActivated(True)

            # create a container for 3D vis of selected map
            case_identifier = os.path.basename(ls_dataset.surface_dataset_path)[:-8]
            try:
                translation = self.vessel_translations[case_identifier]
                rotation = self.vessel_rotations[case_identifier]
            except:
                print("Warning: No translation/rotation found for", case_identifier)
                translation = None
                rotation = None
            levels = self.getLevels()
            container = LatentSpace3DContainer(
                surface_file_path=ls_dataset.surface_dataset_path,
                active_field_name=self.active_field_name,
                scale_min=levels[0],
                scale_max=levels[1],
                lut=self.vtk_lut,
                stenosis_degree=ls_dataset.stenosis_degree,
                stream_sys_path=ls_dataset.systolic_streamline_path,
                stream_dia_path=ls_dataset.diastolic_streamline_path,
                trans=translation,
                rot=rotation,
                landmarks=[ls_dataset.landmark_ACC, ls_dataset.landmark_ACI, ls_dataset.landmark_ACE]
                )
            if self.link_cam_checkbox.isChecked():
                container.useLinkedCamera(self.active_patient_camera)
            self.comp_patient_containers.append(container)
            self.comp_patient_view.GetRenderWindow().AddRenderer(container.renderer)
        
        else:
            # box is already active -> deactive
            on_screen_id = self.active_map_ids.index(index)
            self.active_map_ids.pop(on_screen_id)
            ls_dataset.is_clicked = False
            ls_dataset.map_view.setActivated(False)

            # remove actor, camera, renderer
            container = self.comp_patient_containers.pop(on_screen_id)
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
            self.latent_space_datasets[id].map_view.setIdText(i+1)

            # update viewports
            row = nr_rows - 1 - int(i / nr_cols) # top -> bottom rows
            col = int(i % nr_cols)
            self.comp_patient_containers[i].renderer.SetViewport(col * w, row * h, (col + 1) * w, (row + 1) * h)
            if nr_maps <= 9:
                self.comp_patient_containers[i].setIdText(i+1, True)
            else:
                self.comp_patient_containers[i].setIdText(i+1, False)

        # render all comparison views
        self.comp_patient_view.GetRenderWindow().Render()

    
    def loadPatient(self, patient_dict):
        self.active_patient_dict = patient_dict
        patient_id = self.active_patient_dict['patient_ID']
        if patient_id == None:
            print("No patient ID could be found.")
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

        if not lumen_path:
            return

        # get stenosis degree of new case
        meta_path = lumen_path[:-9] + "meta.txt"
        stenosis_degree = getMetaInformation(meta_path)
        self.active_patient_text.SetInput(case_identifier + "\n" + str(stenosis_degree) + "% Stenosis")

        if lumen_path:
            self.active_patient_reader_lumen.SetFileName(lumen_path)
            self.active_patient_reader_lumen.Update()
            self.active_patient_renderer.AddActor(self.active_patient_actor_lumen)
            self.active_patient_renderer.AddActor(self.active_patient_text)
        else:
            self.active_patient_renderer.RemoveActor(self.active_patient_actor_lumen)
            self.active_patient_renderer.RemoveActor(self.active_patient_text)
        
        if plaque_path:
            self.active_patient_reader_plaque.SetFileName(plaque_path)
            self.active_patient_reader_plaque.Update()
            self.active_patient_renderer.AddActor(self.active_patient_actor_plaque)
        else:
            self.active_patient_renderer.RemoveActor(self.active_patient_actor_plaque)

        # transform from automatic landmarks
        centerline_path = lumen_path[:-4] + "_centerlines.vtp"
        diameter_profile, bifurcation_index, landmark_ACC, landmark_ACI, landmark_ACE = processCenterline(centerline_path)

        source_points = vtk.vtkPoints()
        source_points.InsertNextPoint(landmark_ACC)
        source_points.InsertNextPoint(landmark_ACI)
        source_points.InsertNextPoint(landmark_ACE)

        transform = vtk.vtkLandmarkTransform()
        transform.SetSourceLandmarks(source_points)
        transform.SetTargetLandmarks(LANDMARK_TARGETS)
        transform.SetModeToRigidBody()

        """
        # transform from rotation/translation
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
        transform = vtk.vtkTransform()
        transform.SetMatrix(transform_mat)
        """

        self.transform_filter_lumen.SetTransform(transform)
        self.transform_filter_plaque.SetTransform(transform)

        # reset camera
        self.active_patient_camera.SetPosition(0, 0, -100)
        self.active_patient_camera.SetFocalPoint(0, 0, 0)
        self.active_patient_camera.SetViewUp(0, -1, 0)
        self.active_patient_renderer.ResetCamera()
        self.active_patient_view.GetRenderWindow().Render()

        #########################################
        # sort the latent space
        #########################################

        try:
            col = self.dist_mat_identifiers.index(case_identifier)
        except:
            col = -1

        # make comparisons
        for ls_dataset in self.latent_space_datasets:
            # find distance if it exists
            try:
                row = self.dist_mat_identifiers.index(ls_dataset.case_identifier)
            except:
                row = -1
            if col != -1 and row != -1:
                d = self.dist_mat[row, col]
            else:
                d = 1.0
            # compare
            ls_dataset.compareWithCase(
                distance_to_case=d,
                case_diameter_profile=diameter_profile,
                case_bifurcation_index=bifurcation_index,
                case_stenosis_degree=stenosis_degree)
        self.sortLatentSpace()

    
    def sortLatentSpace(self):
        # update the score values
        shape_w    = 1 if self.shape_checkbox.checkbox.isChecked() else 0
        diam_w     = 1 if self.diameter_checkbox.checkbox.isChecked() else 0
        stenosis_w = 1 if self.stenosis_checkbox.checkbox.isChecked() else 0
        for ls_dataset in self.latent_space_datasets:
            ls_dataset.updateSimilarityScore(shape_weight=shape_w, diam_weight=diam_w, stenosis_weight=stenosis_w)

        # sort the internal datasets
        self.latent_space_datasets = sorted(self.latent_space_datasets, reverse=True)

        # update the ids
        self.active_map_ids.clear()
        for index, ls_dataset in enumerate(self.latent_space_datasets):
            ls_dataset.map_view.list_index = index # will be returned on click
            if ls_dataset.is_clicked:
                self.active_map_ids.append(index)

        # display maps in new order
        self.fillMapView(force_reload=True)
            
    
    def getLevels(self):
        return (self.map_scale_min[self.active_field_name], self.map_scale_max[self.active_field_name])


    def setLevelsMin(self, minval):
        self.map_scale_min[self.active_field_name] = minval
        self.__updateLevels()


    def setLevelsMax(self, maxval):
        self.map_scale_max[self.active_field_name] = maxval
        self.__updateLevels()

    
    def __updateLevels(self):
        levels = self.getLevels()
        self.color_bar.setLevels(levels)
        self.levels_max_spinbox.setRange(levels[0], 999)
        self.levels_min_spinbox.setRange(-999, levels[1])

        # update levels on surface/streamlines if a surface map is displayed
        if not 'velocity' in self.active_field_name:
            for ls_dataset in self.latent_space_datasets:
                ls_dataset.map_view.img.setLevels(levels)
            for container in self.comp_patient_containers:
                container.surface_mapper.SetScalarRange(levels)
        else:
            for container in self.comp_patient_containers:
                container.streamlines_mapper.SetScalarRange(levels)
        self.comp_patient_view.GetRenderWindow().Render()

    
    def setScalarField(self, index):
        self.active_field_name = MAP_FIELD_NAMES[index]
        self.colormap_title.setText(MAP_TITLE_NAMES[index])
        levels = self.getLevels()

        # update labels, colorbar is automatically updated
        self.levels_min_spinbox.setValue(levels[0])
        self.levels_min_spinbox.setRange(-999, levels[1])
        self.levels_max_spinbox.setValue(levels[1])
        self.levels_max_spinbox.setRange(levels[0], 999)

        # velocity field? -> display streamlines
        if 'velocity' in self.active_field_name:
            # unicolor maps with lowest color (velocity 0), display streamlines
            for ls_dataset in self.latent_space_datasets:
                ls_dataset.map_view.img.setLevels((10000, 10000))
            for container in self.comp_patient_containers:
                container.surface_mapper.SetScalarRange(10000, 10000)
                if 'sys' in self.active_field_name:
                    container.setViewStreamlines(levels, systolic=True)
                else:
                    container.setViewStreamlines(levels, systolic=False)
            self.levels_min_spinbox.setSingleStep(0.5)
            self.levels_max_spinbox.setSingleStep(0.5)
            self.levels_min_spinbox.setSuffix(" [m/s]")
            self.levels_max_spinbox.setSuffix(" [m/s]")

        # surface field? -> display surface
        else:
            # set field on 3D views
            for container in self.comp_patient_containers:
                container.setViewSurface(self.active_field_name, levels)

            # set field on map views
            for ls_dataset in self.latent_space_datasets:
                img = ls_dataset.map_view.img
                img.setImage(ls_dataset.map_dataset[self.active_field_name])
                img.setLevels(levels)
            self.levels_min_spinbox.setSingleStep(10)
            self.levels_max_spinbox.setSingleStep(10)
            self.levels_min_spinbox.setSuffix(" [Pa]")
            self.levels_max_spinbox.setSuffix(" [Pa]")
        
        # update 3D views
        self.comp_patient_view.GetRenderWindow().Render()

    
    def setColorMap(self, index):
        self.cmap = pg.colormap.get(COLORMAP_KEYS[index])
        self.vtk_lut = getVTKLookupTable(self.cmap)
        self.color_bar.setColorMap(self.cmap)
        for ls_dataset in self.latent_space_datasets:
            ls_dataset.map_view.img.setColorMap(self.cmap)
        for container in self.comp_patient_containers:
            container.surface_mapper.SetLookupTable(self.vtk_lut)
            container.streamlines_mapper.SetLookupTable(self.vtk_lut)
        self.comp_patient_view.GetRenderWindow().Render()
        
    
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

    
    def resetCompViews(self):
        # delete active 3D views of selected maps
        for id in self.active_map_ids:
            self.latent_space_datasets[id].is_clicked = False
            self.latent_space_datasets[id].map_view.setActivated(False)
        self.active_map_ids.clear()
        for c in self.comp_patient_containers:
            self.comp_patient_view.GetRenderWindow().RemoveRenderer(c.renderer)
        self.comp_patient_containers.clear()
        self.comp_patient_view.GetRenderWindow().Render()


    def resetAllViews(self):
        # remove active patient
        self.active_patient_renderer.RemoveActor(self.active_patient_actor_lumen)
        self.active_patient_renderer.RemoveActor(self.active_patient_actor_plaque)
        self.active_patient_view.GetRenderWindow().Render()

        # remove comparison views
        self.resetCompViews()

        # remove map views from latent space navigator but keep map cache
        self.latent_space_widget.clear()


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



class MapViewBox(pg.ViewBox):
    clicked = pyqtSignal(int)
    def __init__(self, list_index, img_data, levels, colormap, parent=None):
        super().__init__(parent=parent, border=None, lockAspect=True, enableMouse=False, enableMenu=False, defaultPadding=0)
        # needs to know the global index, as it is returned on click
        self.list_index = list_index

        # create internal items
        self.bar_stack = BarStackItem(img_data.shape[0], img_data.shape[1], [1.0, 1.0, 1.0], thickness=50.0)
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


    def updateScoreBars(self, normalized_values_list):
        self.bar_stack.setValues(normalized_values_list)


    def mousePressEvent(self, ev):
        self.clicked.emit(self.list_index)

    
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
    def __init__(self, y_offset, x_width, normalized_values_list, thickness):
        super().__init__()
        self.y_offset = y_offset
        self.x_width = x_width
        self.ends = np.array(normalized_values_list)[::-1] * self.x_width
        self.thickness = thickness
        self.colors = [HTML_STENOSIS_COLOR, HTML_DIAMETER_COLOR, HTML_SHAPE_COLOR] # colors are internally ordered bottom->top
        self.generatePicture()

    
    def setValues(self, normalized_values_list):
        self.ends = np.array(normalized_values_list)[::-1] * self.x_width
        self.generatePicture()
    

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        p.setPen(pg.mkPen(None))
        for i in range(len(self.ends)):
            p.setBrush(pg.mkBrush(self.colors[i]))
            p.drawRect(QtCore.QRectF(0.0, self.y_offset + self.thickness*(i+1), self.ends[i], self.thickness))
        p.setPen(pg.mkPen(0,0,0))
        p.drawRect(QtCore.QRectF(0.0, self.y_offset + self.thickness, self.x_width, 0.0))
        p.drawRect(QtCore.QRectF(0.0, self.y_offset + self.thickness, 0.0, self.thickness * 3))
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
    def __init__(self, surface_file_path, active_field_name, scale_min, scale_max, lut, stenosis_degree,
                 stream_sys_path=None, stream_dia_path=None, trans=None, rot=None, landmarks=None):
        # id text
        self.stenosis_degree_str = "\n" + str(stenosis_degree) + "% Stenosis"
        self.identifier_text = vtk.vtkTextActor()
        p = self.identifier_text.GetTextProperty()
        p.SetColor(0, 0, 0)
        p.SetFontSize(20)
        p.SetBackgroundColor(1, 1, 1)

        if landmarks is None and trans is not None and rot is not None:
            # create transform for 3D objects from translation/rotation
            rotation_mat = np.eye(4)
            if rot is not None:
                rotation_mat[0:3, 0:3] = np.reshape(rot, (3,3), order='F') # matrix is given column-wise
            translation_mat = np.eye(4)
            if trans is not None:
                translation_mat[0:3, 3] = np.array(trans)
            transform_mat = rotation_mat @ translation_mat # translate, then rotate
            transform_mat = transform_mat.flatten(order='C') # row-wise input for vtk
            transform = vtk.vtkTransform()
            transform.SetMatrix(transform_mat)
        elif landmarks is not None:
            # create transform for 3D objects from landmarks (preferred)
            source_points = vtk.vtkPoints()
            source_points.InsertNextPoint(landmarks[0]) # ACC
            source_points.InsertNextPoint(landmarks[1]) # ACI
            source_points.InsertNextPoint(landmarks[2]) # ACE

            transform = vtk.vtkLandmarkTransform()
            transform.SetSourceLandmarks(source_points)
            transform.SetTargetLandmarks(LANDMARK_TARGETS)
            transform.SetModeToRigidBody()
        else:
            # no transform
            transform = vtk.vtkTransform()

        # surface
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(surface_file_path)
        geometry_filter = vtk.vtkGeometryFilter() # unstructured grid -> polydata
        geometry_filter.SetInputConnection(reader.GetOutputPort())
        normals = vtk.vtkPolyDataNormals()
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.ConsistencyOn()
        normals.SplittingOff()
        normals.SetInputConnection(geometry_filter.GetOutputPort())
        normals.Update()
        self.surface = normals.GetOutput()
        self.surface.GetPointData().SetActiveScalars(active_field_name)
        transform_filter = vtk.vtkTransformFilter()
        transform_filter.SetInputData(self.surface)
        transform_filter.SetTransform(transform)
        self.surface_mapper = vtk.vtkDataSetMapper()
        self.surface_mapper.SetInputConnection(transform_filter.GetOutputPort())
        self.surface_mapper.SetScalarRange(scale_min, scale_max)
        self.surface_mapper.SetLookupTable(lut)
        self.surface_actor = vtk.vtkActor()
        self.surface_actor.GetProperty().SetInterpolationToGouraud()
        self.surface_actor.SetMapper(self.surface_mapper)
        
        # streamlines
        if stream_sys_path is not None:
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(stream_sys_path)
            reader.Update()
            self.streamlines_systolic = reader.GetOutput()
            self.streamlines_systolic.GetPointData().SetActiveScalars('velocity_systolic_mag')
        else:
            self.streamlines_systolic = vtk.vtkPolyData()

        if stream_dia_path is not None:
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(stream_dia_path)
            reader.Update()
            self.streamlines_diastolic = reader.GetOutput()
            self.streamlines_diastolic.GetPointData().SetActiveScalars('velocity_diastolic_mag')
        else:
            self.streamlines_diastolic = vtk.vtkPolyData()

        self.streamlines_transform_filter = vtk.vtkTransformFilter()
        self.streamlines_transform_filter.SetInputData(self.streamlines_systolic)
        self.streamlines_transform_filter.SetTransform(transform)
        self.streamlines_mapper = vtk.vtkDataSetMapper()
        self.streamlines_mapper.SetInputConnection(self.streamlines_transform_filter.GetOutputPort())
        self.streamlines_mapper.SetScalarRange(scale_min, scale_max)
        self.streamlines_mapper.SetLookupTable(lut)
        self.streamlines_actor = vtk.vtkActor()
        self.streamlines_actor.GetProperty().SetLineWidth(3)
        self.streamlines_actor.SetMapper(self.streamlines_mapper)

        # create a renderer, save own camera (viewports will be set later)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1, 1, 1)
        self.camera = self.renderer.GetActiveCamera()
        self.camera.SetPosition(0, 0, -100)
        self.camera.SetFocalPoint(0, 0, 0)
        self.camera.SetViewUp(0, -1, 0)
        if 'velocity_sys' in active_field_name:
            self.renderer.AddActor(self.streamlines_actor)
            self.surface_actor.GetProperty().FrontfaceCullingOn()
        elif 'velocity_dia' in active_field_name:
            self.streamlines_transform_filter.SetInputData(self.streamlines_diastolic)
            self.renderer.AddActor(self.streamlines_actor)
            self.surface_actor.GetProperty().FrontfaceCullingOn()
        self.renderer.AddActor(self.surface_actor)
        self.renderer.AddActor(self.identifier_text)
        self.renderer.ResetCamera()
    
    def setIdText(self, identifier, show_degree=True):
        if show_degree:
            self.identifier_text.SetInput(str(identifier) + self.stenosis_degree_str)
        else:
            self.identifier_text.SetInput(str(identifier))
    
    def setViewStreamlines(self, levels, systolic:bool):
        if systolic:
            self.streamlines_transform_filter.SetInputData(self.streamlines_systolic)
        else:
            self.streamlines_transform_filter.SetInputData(self.streamlines_diastolic)

        self.surface.GetPointData().SetActiveScalars(" ") # disables scalar mapping on surface
        self.surface_actor.GetProperty().FrontfaceCullingOn()
        self.streamlines_transform_filter.Update()
        self.streamlines_mapper.SetScalarRange(levels)
        self.renderer.AddActor(self.streamlines_actor)

    def setViewSurface(self, field_name, levels):
        self.surface.GetPointData().SetActiveScalars(field_name)
        self.surface_mapper.SetScalarRange(levels)
        self.surface_actor.GetProperty().FrontfaceCullingOff()
        self.renderer.RemoveActor(self.streamlines_actor)

    def useLinkedCamera(self, cam):
        self.renderer.SetActiveCamera(cam)

    def useOwnCamera(self):
        # keep current view, reset to full model
        c = self.renderer.GetActiveCamera()
        self.camera.SetPosition(c.GetPosition())
        self.camera.SetFocalPoint(c.GetFocalPoint())
        self.camera.SetViewUp(c.GetViewUp())
        self.renderer.SetActiveCamera(self.camera)
        self.renderer.ResetCamera()



from PyQt5.QtWidgets import QFrame
class VerticalLine(QFrame):
    """
    A vertical line for use in Qt Layouts.
    """
    def __init__(self):
        super(VerticalLine, self).__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)



class CheckBoxDecorator(QWidget):
    def __init__(self, text, color):
        super(CheckBoxDecorator, self).__init__()
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(True)
        self.label = QLabel('<font color="#FFFFFF">'+text+'</font>')
        l = QHBoxLayout()
        l.addWidget(self.checkbox)
        l.addWidget(self.label)
        l.addStretch(1)
        self.setLayout(l)
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(color))
        self.setAutoFillBackground(True)
        self.setPalette(pal)
        self.setMinimumWidth(191) # ensures same width on all boxes