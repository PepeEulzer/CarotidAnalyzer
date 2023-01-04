import os
from collections import OrderedDict
import time

import numpy as np
import nrrd
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QPushButton

from defaults import *
from modules.Interactors import ImageSliceInteractor, VolumeRenderingInteractor

class CropModule(QWidget):
    """
    Module for cropping the left/right carotid from a full CTA volume.
    """
    data_modified = pyqtSignal()
    new_left_volume = pyqtSignal()
    new_right_volume = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.crop_image_left = None
        self.crop_image_right = None
        
        self.box_left_source = vtk.vtkCubeSource()
        self.box_left_mapper = vtk.vtkPolyDataMapper()
        self.box_left_mapper.SetInputConnection(self.box_left_source.GetOutputPort())
        self.box_left_actor = vtk.vtkActor()
        self.box_left_actor.GetProperty().ShadingOff()
        self.box_left_actor.GetProperty().SetColor(COLOR_LEFT)
        self.box_left_actor.GetProperty().SetRepresentationToWireframe()
        self.box_left_actor.GetProperty().SetLineWidth(5.0)
        self.box_left_actor.GetProperty().SetAmbient(1.0)
        self.box_left_actor.GetProperty().SetDiffuse(0.0)
        self.box_left_actor.SetMapper(self.box_left_mapper)

        self.box_right_source = vtk.vtkCubeSource()
        self.box_right_mapper = vtk.vtkPolyDataMapper()
        self.box_right_mapper.SetInputConnection(self.box_right_source.GetOutputPort())
        self.box_right_actor = vtk.vtkActor()
        self.box_right_actor.GetProperty().ShadingOff()
        self.box_right_actor.GetProperty().SetColor(COLOR_RIGHT)
        self.box_right_actor.GetProperty().SetRepresentationToWireframe()
        self.box_right_actor.GetProperty().SetLineWidth(5.0)
        self.box_right_actor.GetProperty().SetAmbient(1.0)
        self.box_right_actor.GetProperty().SetDiffuse(0.0)
        self.box_right_actor.SetMapper(self.box_left_mapper)
        self.box_right_actor.SetMapper(self.box_right_mapper)

        self.box_selection_source = vtk.vtkCubeSource()
        self.box_selection_mapper = vtk.vtkPolyDataMapper()
        self.box_selection_mapper.SetInputConnection(self.box_selection_source.GetOutputPort())
        self.box_selection_actor = vtk.vtkActor()
        self.box_selection_actor.GetProperty().ShadingOff()
        self.box_selection_actor.GetProperty().SetRepresentationToWireframe()
        self.box_selection_actor.GetProperty().SetLineWidth(2.0)
        self.box_selection_actor.GetProperty().SetAmbient(1.0)
        self.box_selection_actor.GetProperty().SetDiffuse(0.0)
        self.box_selection_actor.SetMapper(self.box_selection_mapper)
        self.box_selection_actor.SetMapper(self.box_selection_mapper)

        # prop picker for clicking on image
        self.picker = vtk.vtkPropPicker()
        
        self.slice_view = ImageSliceInteractor(self)
        self.volume_view = VolumeRenderingInteractor(self)
        self.slice_view_slider = QSlider(Qt.Horizontal)
        self.button_set_left = QPushButton("Set Left Volume")
        self.button_set_left.setStyleSheet("color:" + COLOR_LEFT_HEX)
        self.button_set_left.setCheckable(True)
        self.button_set_left.setEnabled(False)
        self.button_set_right = QPushButton("Set Right Volume")
        self.button_set_right.setStyleSheet("color:" + COLOR_RIGHT_HEX)
        self.button_set_right.setCheckable(True)
        self.button_set_right.setEnabled(False)

        self.cut_left_actor = self.__getCutActor(self.box_left_source.GetOutputPort(), COLOR_LEFT)
        self.cut_right_actor = self.__getCutActor(self.box_right_source.GetOutputPort(), COLOR_RIGHT)

        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.button_set_right)
        self.button_layout.addWidget(self.button_set_left)
        self.slice_view_layout = QVBoxLayout()
        self.slice_view_layout.addLayout(self.button_layout)
        self.slice_view_layout.addWidget(self.slice_view)
        self.slice_view_layout.addWidget(self.slice_view_slider)
        self.top_layout = QHBoxLayout(self)
        self.top_layout.addLayout(self.slice_view_layout)
        self.top_layout.addWidget(self.volume_view)

        # connect signals/slots
        self.slice_view.slice_changed[int].connect(self.sliceChanged)
        self.slice_view_slider.valueChanged[int].connect(self.slice_view.setSlice)
        self.button_set_left.clicked[bool].connect(self.setLeftVolume)
        self.button_set_right.clicked[bool].connect(self.setRightVolume)

        # initialize VTK
        self.slice_view.Initialize()
        self.slice_view.Start()
        self.volume_view.Initialize()
        self.volume_view.Start()

    
    def __getCutActor(self, output_port, color):
        cutter = vtk.vtkCutter()
        cutter.SetInputConnection(output_port)
        cutter.SetCutFunction(self.slice_view.image_mapper.GetSlicePlane())
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cutter.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetLineWidth(3.0)
        actor.GetProperty().SetAmbient(1.0)
        actor.GetProperty().SetDiffuse(0.0)
        return actor

    
    def sliceChanged(self, slice_nr):
        self.slice_view_slider.setSliderPosition(slice_nr)
        self.volume_view.Render()

    
    def pickWorldPosition(self, obj, event):
        # pick current mouse position, screen coordinates
        x_view, y_view = self.slice_view.GetEventPosition()  

        # world coordinates 
        self.picker.Pick(x_view, y_view, self.slice_view.slice, self.slice_view.renderer) 
        x,y,z = self.picker.GetPickPosition()

        # move selection box
        xs, ys, zs = self.crop_volume_size
        self.box_selection_source.SetBounds(x - xs, x + xs,
                                            y - ys, y + ys,
                                            z - zs, z + zs)

        # update scenes
        self.slice_view.GetRenderWindow().Render()
        self.volume_view.GetRenderWindow().Render()


    def __activateHoverVolume(self, color, fin_func):
        self.box_selection_actor.GetProperty().SetColor(color)
        self.slice_view.renderer.AddActor(self.box_selection_actor)
        self.volume_view.renderer.AddActor(self.box_selection_actor)
        self.mouse_move_observer = self.slice_view.interactor_style.AddObserver("MouseMoveEvent", self.pickWorldPosition)
        self.left_click_observer = self.slice_view.interactor_style.AddObserver("LeftButtonPressEvent", fin_func)


    def __disableHoverVolume(self):
        self.slice_view.interactor_style.RemoveObserver(self.mouse_move_observer)
        self.slice_view.interactor_style.RemoveObserver(self.left_click_observer)
        self.slice_view.renderer.RemoveActor(self.box_selection_actor)
        self.volume_view.renderer.RemoveActor(self.box_selection_actor)
        self.slice_view.GetRenderWindow().Render()
        self.volume_view.GetRenderWindow().Render()


    def setLeftVolume(self, activated):
        if activated:
            self.__activateHoverVolume(COLOR_LEFT_LIGHT, self.setLeftVolumeFinished)
        else:
            self.__disableHoverVolume()


    def setRightVolume(self, activated):
        if activated:
            self.__activateHoverVolume(COLOR_RIGHT_LIGHT, self.setRightVolumeFinished)
        else:
            self.__disableHoverVolume()

    
    def __setVolumeFinished(self, box_source, box_actor, cut_actor, left=True):
        # pick current mouse position, screen coordinates
        x_view, y_view = self.slice_view.GetEventPosition()  

        # world coordinates 
        self.picker.Pick(x_view, y_view, self.slice_view.slice, self.slice_view.renderer) 
        pos = self.picker.GetPickPosition()

        # discrete image coordinates
        origin = self.image.GetOrigin()
        spacing = self.image.GetSpacing()
        x,y,z = (int(round((pos[0] - origin[0]) / spacing[0])), 
                 int(round((pos[1] - origin[1]) / spacing[1])), 
                 int(self.slice_view.slice))

        # crop volume around center +- 30, 36, 62
        extractor = vtk.vtkExtractVOI()
        extractor.SetInputData(self.image)
        extractor.SetVOI(x-29, x+30, y-35, y+36, z-61, z+62)
        
        # scale resolution * 2
        reslicer = vtk.vtkImageReslice()
        reslicer.SetInputConnection(extractor.GetOutputPort())
        reslicer.SetInterpolationModeToCubic()
        reslicer.SetOutputExtent(0, 119, 0, 143, 0, 247)
        reslicer.SetOutputSpacing([s*0.5 for s in spacing])
        reslicer.Update()
        crop_image = reslicer.GetOutput()

        # adapt crop display
        ox, oy, oz = crop_image.GetOrigin()
        sx, sy, sz = crop_image.GetSpacing()
        x, y, z = crop_image.GetDimensions()
        box_source.SetBounds(ox, ox + sx*x,
                             oy, oy + sy*y,
                             oz, oz + sz*z)
        self.volume_view.renderer.AddActor(box_actor)
        self.volume_view.renderer.AddActor(cut_actor)
        self.slice_view.renderer.AddActor(cut_actor)

        # reset observers and scene
        self.slice_view.interactor_style.RemoveObserver(self.mouse_move_observer)
        self.slice_view.interactor_style.RemoveObserver(self.left_click_observer)
        self.slice_view.renderer.RemoveActor(self.box_selection_actor)
        self.volume_view.renderer.RemoveActor(self.box_selection_actor)
        self.slice_view.GetRenderWindow().Render()
        self.volume_view.GetRenderWindow().Render()

        # emit data modified
        if left:
            self.crop_image_left = crop_image
        else:
            self.crop_image_right = crop_image
        self.data_modified.emit()


    def setLeftVolumeFinished(self, obj, event):
        self.__setVolumeFinished(self.box_left_source, self.box_left_actor, self.cut_left_actor, left=True)
        self.button_set_left.setChecked(False)


    def setRightVolumeFinished(self, obj, event):
        self.__setVolumeFinished(self.box_right_source, self.box_right_actor, self.cut_right_actor, left=False)
        self.button_set_right.setChecked(False)  


    def showEvent(self, event):
        self.slice_view.Enable()
        self.slice_view.EnableRenderOn()
        self.volume_view.Enable()
        self.volume_view.EnableRenderOn()
        super(CropModule, self).showEvent(event)


    def hideEvent(self, event):
        self.slice_view.Disable()
        self.slice_view.EnableRenderOff()
        self.volume_view.Disable()
        self.volume_view.EnableRenderOff()
        super(CropModule, self).hideEvent(event)


    def __loadCropVolumeBox(self, filename, box_source, box_actor, cut_actor):
        if filename:
            header = nrrd.read_header(filename)
            ox, oy, oz = header['space origin']
            sx, sy, sz = np.diagonal(header['space directions'])
            x, y, z = header['sizes']
            if sx < 0: # mirrored x/y axes
                box_source.SetBounds(
                         ox + sx*x, ox,
                         oy + sy*y, oy,
                         oz, oz + sz*z
                )
            else:
                box_source.SetBounds(
                            ox, ox + sx*x,
                            oy, oy + sy*y,
                            oz, oz + sz*z
                )
            self.volume_view.renderer.AddActor(box_actor)
            self.volume_view.renderer.AddActor(cut_actor)
            self.slice_view.renderer.AddActor(cut_actor)
        else:
            self.volume_view.renderer.RemoveActor(box_actor)
            self.volume_view.renderer.RemoveActor(cut_actor)
            self.slice_view.renderer.RemoveActor(cut_actor)

    
    def resetViews(self):
        self.button_set_left.setEnabled(False)
        self.button_set_right.setEnabled(False)
        self.volume_view.renderer.RemoveActor(self.box_left_actor)
        self.volume_view.renderer.RemoveActor(self.cut_left_actor)
        self.slice_view.renderer.RemoveActor(self.cut_left_actor)
        self.volume_view.renderer.RemoveActor(self.box_right_actor)
        self.volume_view.renderer.RemoveActor(self.cut_right_actor)
        self.slice_view.renderer.RemoveActor(self.cut_right_actor)
        self.slice_view.reset()
        self.volume_view.reset()


    def loadPatient(self, patient_dict,image=None):
        self.patient_dict = patient_dict

        # load patient if data from dicom
        if image:  
            self.image = image
            sx, sy, sz = self.image.GetSpacing()
           
        # load patient if data from nrrd   
        else: 
            if not patient_dict['volume_raw']:
                self.image = None
                self.resetViews()
                return

            reader = vtk.vtkNrrdReader()
            reader.SetFileName(patient_dict['volume_raw'])
            reader.Update()
            self.image = reader.GetOutput()

        # compute crop volume size around a center
        # needs to be 1/4 of target dimension (120 144 248)
        self.crop_volume_size = (30*sx, 36*sy, 62*sz)

        # set the volume image in both views
        self.volume_view.setImage(self.image)
        self.slice_view.setImage(self.image)
        if image: 
            self.slice_view.text_patient.SetInput(patient_dict['patient_ID'])
        else:
            self.slice_view.text_patient.SetInput(os.path.basename(patient_dict['volume_raw'])[:-5])
        self.slice_view_slider.setRange(
            self.slice_view.min_slice,
            self.slice_view.max_slice
        )

        # load crop volume boxes if they exist
        self.__loadCropVolumeBox(self.patient_dict['volume_left'], self.box_left_source, self.box_left_actor, self.cut_left_actor)
        self.__loadCropVolumeBox(self.patient_dict['volume_right'], self.box_right_source, self.box_right_actor, self.cut_right_actor)

        self.slice_view_slider.setSliderPosition(self.slice_view.slice)
        # enable edit options
        self.button_set_left.setEnabled(True)
        self.button_set_right.setEnabled(True)

    
    def saveVolumeNrrd(self, volume, path):
        sx, sy, sz = volume.GetSpacing()
        ox, oy, oz = volume.GetOrigin()
        x_dim, y_dim, z_dim = volume.GetDimensions()
        header = OrderedDict()
        header['dimension'] = 3
        header['space'] = 'left-posterior-superior'
        header['space directions'] = [[sx, 0, 0], [0, sy, 0], [0, 0, sz]]
        header['kinds'] = ['domain', 'domain', 'domain']
        header['endian'] = 'little'
        header['encoding'] = 'gzip'
        header['space origin'] = [ox, oy, oz]
        segmentation = vtk_to_numpy(volume.GetPointData().GetScalars()).astype(np.int16)
        segmentation = segmentation.reshape(x_dim, y_dim, z_dim, order='F')
        nrrd.write(path, segmentation, header)

    
    def save(self):
        patient_ID = self.patient_dict['patient_ID']
        base_path  = self.patient_dict['base_path']

        if self.crop_image_left is not None:
            path_left = os.path.join(base_path, patient_ID + "_left.nrrd")
            self.saveVolumeNrrd(self.crop_image_left, path_left)
            self.new_left_volume.emit()
            self.crop_image_left = None

        if self.crop_image_right is not None:
            path_right = os.path.join(base_path, patient_ID + "_right.nrrd")
            self.saveVolumeNrrd(self.crop_image_right, path_right)
            self.new_right_volume.emit()
            self.crop_image_right = None

    
    def discard(self):
        self.crop_image_left = None
        self.crop_image_right = None
        self.__loadCropVolumeBox(self.patient_dict['volume_left'], self.box_left_source, self.box_left_actor, self.cut_left_actor)
        self.__loadCropVolumeBox(self.patient_dict['volume_right'], self.box_right_source, self.box_right_actor, self.cut_right_actor)
        self.slice_view.GetRenderWindow().Render()
        self.volume_view.GetRenderWindow().Render()


    def close(self):
        self.slice_view.Finalize()
        self.volume_view.Finalize()