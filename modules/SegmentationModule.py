import os
from collections import OrderedDict

import numpy as np
import nrrd
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import  (
    QWidget, QVBoxLayout, QHBoxLayout, QSlider, QTabWidget,
    QPushButton, QMessageBox, QGridLayout, QLabel, QToolBar, QAction, QSizePolicy
)

from modules.Interactors import ImageSliceInteractor, IsosurfaceInteractor
from defaults import *

class SegmentationModuleTab(QWidget):  
    """
    Tab view of a right OR left side carotid for segmentation.
    """
    data_modified = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)

        # state
        self.image = None                # underlying CTA volume image
        self.image_data = None           # numpy array of raw image scalar data
        self.label_map = None            # segmentation label map
        self.label_map_data = None       # numpy array of raw label map scalar data
        self.threshold_img = None        # image to display threshold 
        self.volume_file = False         # path to CTA volume file
        self.pred_file = False           # path to CNN segmentation prediction file
        self.plaque_pending = True       # True if no plaque pixels exist yet
        self.lumen_pending = True        # True if not lumen pixels exist yet
        self.model_camera_pending = True # True if camera of model_view has not been set yet
        self.editing_active = False      # True if label map editing is active
        self.brush_size = 15             # size of brush on label map
        self.threshold = 0               # value of threshold for drawing with brush 
        self.draw3D = False              # dimension of brush (2/3D) 
        self.eraser = False              # use of eraser or brush 

        # on-screen objects
        self.slice_view = ImageSliceInteractor(self)
        self.slice_view_slider = QSlider(Qt.Horizontal)
        self.slice_view_slider_label = QLabel("Slice")
        self.slice_view_slider.setEnabled(False)
        self.slice_view_slider_label.setEnabled(False)

        self.model_view = IsosurfaceInteractor(self)
        self.CNN_button = QPushButton("New Segmentation: Initialize with CNN") # für button in Toolbar: self.toolbarCNN  

        # define sliders
        self.brush_size_slider = QSlider(Qt.Horizontal)  
        self.brush_size_slider.setMinimum(1)
        self.brush_size_slider.setMaximum(51)
        self.brush_size_slider.setSingleStep(2)
        self.brush_size_slider.setValue(self.brush_size)
        self.brush_size_slider.setTickInterval(1)
        self.brush_slider_label = QLabel("Brush/Eraser Size         ")
        self.brush_size_slider.setEnabled(False)
        self.brush_slider_label.setEnabled(False) 
        self.threshold_slider = QSlider(Qt.Horizontal) 
        self.threshold_slider.setSingleStep(1)  
        self.threshold_slider.setTickInterval(1)  
        self.threshold_slider_label = QLabel("Threshold: "+ str(self.threshold) + " (HU)")
        self.threshold_slider.setEnabled(False)
        self.threshold_slider_label.setEnabled(False)
        
        # add sliders to grid
        self.slider_layout = QGridLayout()
        self.slider_layout.addWidget(self.brush_slider_label, 0,0,1,2)
        self.slider_layout.addWidget(self.threshold_slider_label,1,0,1,2)
        self.slider_layout.addWidget(self.slice_view_slider_label, 2,0,1,2)
        self.slider_layout.setColumnStretch(2, 4)
        self.slider_layout.addWidget(self.brush_size_slider, 0,2)
        self.slider_layout.addWidget(self.threshold_slider,1,2)
        self.slider_layout.addWidget(self.slice_view_slider,2,2)

        # actions for toolbar
        #self.toolbar_CNN = QAction("New Segmentation: Initialize with CNN")
        self.toolbar_edit = QAction("Edit Segmenation", self)
        self.toolbar_edit.setEnabled(False)
        self.toolbar_edit.setCheckable(True)
        self.toolbar_lumen = QAction("Lumen", self)
        self.toolbar_lumen.setCheckable(True)
        self.toolbar_lumen.setEnabled(False)
        self.toolbar_plaque = QAction("Plaque", self)
        self.toolbar_plaque.setCheckable(True)
        self.toolbar_plaque.setEnabled(False)
        self.toolbar_brush2D = QAction("2D", self)
        self.toolbar_brush2D.setCheckable(True)
        self.toolbar_brush2D.setEnabled(False)
        self.toolbar_brush3D = QAction("3D", self)
        self.toolbar_brush3D.setCheckable(True)
        self.toolbar_brush3D.setEnabled(False)
        self.toolbar_auto_update = QAction("auto-update 3D model")
        self.toolbar_auto_update.setCheckable(True)
        self.toolbar_auto_update.setEnabled(False)
        spacer1 = QWidget()
        spacer1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        spacer2 = QWidget()
        spacer2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # set toolbar
        self.edit_toolbar = QToolBar()
        self.edit_toolbar.addAction(self.toolbar_edit)
        self.edit_toolbar.addWidget(spacer1)
        self.edit_toolbar.addAction(self.toolbar_lumen)
        self.edit_toolbar.addAction(self.toolbar_plaque)
        self.edit_toolbar.addSeparator()
        self.edit_toolbar.addAction(self.toolbar_brush2D)
        self.edit_toolbar.addAction(self.toolbar_brush3D)
        self.edit_toolbar.addWidget(spacer2)
        self.edit_toolbar.addAction(self.toolbar_auto_update)


        # add everything to a layou
        self.slice_view_layout = QVBoxLayout()
        self.slice_view_layout.addWidget(self.CNN_button)
        self.slice_view_layout.addWidget(self.edit_toolbar)
        self.slice_view_layout.addLayout(self.slider_layout)
        self.slice_view_layout.addWidget(self.slice_view)

        self.top_layout = QHBoxLayout(self)
        self.top_layout.addLayout(self.slice_view_layout)
        self.top_layout.addWidget(self.model_view)
        

        # vtk objects
        self.lumen_outline_actor3D, self.lumen_outline_actor2D = self.__createOutlineActors(
            self.model_view.smoother_lumen.GetOutputPort(), COLOR_LUMEN_DARK, COLOR_LUMEN)
        self.plaque_outline_actor3D, self.plaque_outline_actor2D = self.__createOutlineActors(
            self.model_view.smoother_plaque.GetOutputPort(), COLOR_PLAQUE_DARK, COLOR_PLAQUE)
        self.__setupLUT()  # setup lookup table to display masks and threshold 
        self.__setupEditingPipeline()


        # connect signals/slots
        self.CNN_button.pressed.connect(self.generateCNNSeg)
        self.slice_view.slice_changed[int].connect(self.sliceChanged)
        self.slice_view_slider.valueChanged[int].connect(self.slice_view.setSlice)
        self.toolbar_edit.triggered[bool].connect(self.edit)
        self.toolbar_brush2D.triggered[bool].connect(self.set2DBrush)
        self.toolbar_brush3D.triggered[bool].connect(self.set3DBrush)
        self.brush_size_slider.valueChanged[int].connect(self.brushSizeChanged)
        self.threshold_slider.valueChanged[int].connect(self.thresholdChanged)
        self.threshold_slider.sliderPressed.connect(self.showThreshold)
        self.threshold_slider.sliderReleased.connect(self.hideThreshold)
        self.toolbar_lumen.triggered[bool].connect(self.setColorLumen) 
        self.toolbar_plaque.triggered[bool].connect(self.setColorPlaque)

        # initialize VTK
        self.slice_view.Initialize()
        self.slice_view.Start()
        self.model_view.Initialize()
        self.model_view.Start()


    def __createOutlineActors(self, output_port, color3D, color2D):
        cutter = vtk.vtkCutter()
        cutter.SetInputConnection(output_port)
        cutter.SetCutFunction(self.slice_view.image_mapper.GetSlicePlane())
        mapper = vtk.vtkPolyDataMapper()
        mapper.ScalarVisibilityOff()
        mapper.SetInputConnection(cutter.GetOutputPort())
        actor3D = vtk.vtkActor()
        actor3D.SetMapper(mapper)
        actor3D.GetProperty().SetColor(color3D)
        actor3D.GetProperty().SetLineWidth(5)
        actor3D.GetProperty().RenderLinesAsTubesOn()
        actor2D = vtk.vtkActor()
        actor2D.SetMapper(mapper)
        actor2D.GetProperty().SetColor(color2D)
        actor2D.GetProperty().SetLineWidth(2)
        return actor3D, actor2D


    def __setupLUT(self):
        # lookup table for label map 
        self.lut_lm = vtk.vtkLookupTable()  
        self.lut_lm.SetNumberOfTableValues(3)
        self.lut_lm.SetTableRange(0,2)
        self.lut_lm.SetTableValue(0, 0.0, 0.0, 0.0, 0.0)  # set color of backround (id 0) to black with transparency 0
        alpha = (0.5,)
        self.lumen_rgba = COLOR_LUMEN + alpha
        self.plaque_rgba = COLOR_PLAQUE + alpha  
        self.lut_lm.SetTableValue(1, self.plaque_rgba)
        self.lut_lm.SetTableValue(2, self.lumen_rgba)
        self.lut_lm.Build() 

        # lookup table for display of threshold 
        self.lut_threshold = vtk.vtkLookupTable()
        self.lut_threshold.SetNumberOfTableValues(2)
        self.lut_threshold.SetTableRange(0,1)
        self.lut_threshold.SetTableValue(0, 0.0, 0.0, 0.0, 0.0)  # set color of backround of mask (id 0) to black with transparency 0
        self.lut_threshold.SetTableValue(1, 0.0,0.0, 1.0, 0.4)  # set color of areas with values above threshold 
        self.lut_threshold.Build()


    def __setupEditingPipeline(self):
        # map 2D display through colormap
        self.masks_color_mapped = vtk.vtkImageMapToColors() 
        self.masks_color_mapped.SetLookupTable(self.lut_lm) 
        self.masks_color_mapped.PassAlphaToOutputOn()
        
        self.mask_slice_mapper = vtk.vtkOpenGLImageSliceMapper()
        self.mask_slice_mapper.SetInputConnection(self.masks_color_mapped.GetOutputPort())
        self.mask_slice_mapper.SetSliceNumber(self.slice_view.slice)
        
        self.mask_slice_actor = vtk.vtkImageActor()
        self.mask_slice_actor.SetMapper(self.mask_slice_mapper)
        self.mask_slice_actor.InterpolateOff()
    
        # circle/sphere around mouse when drawing 
        self.circle = self.setUpCircle()
        circle_mapper = vtk.vtkPolyDataMapper()
        circle_mapper.SetInputConnection(self.circle.GetOutputPort())
        self.circle_actor = vtk.vtkActor()
        self.circle_actor.SetMapper(circle_mapper)

        # map pixels above threshold through colormap
        self.threshold_color_mapped = vtk.vtkImageMapToColors() 
        self.threshold_color_mapped.SetLookupTable(self.lut_threshold) 
        self.threshold_color_mapped.PassAlphaToOutputOn()  
        
        self.threshold_mapper = vtk.vtkOpenGLImageSliceMapper()
        self.threshold_mapper.SetInputConnection(self.threshold_color_mapped.GetOutputPort())
        self.threshold_mapper.SetSliceNumber(self.slice_view.slice)

        self.threshold_actor = vtk.vtkImageActor()
        self.threshold_actor.SetMapper(self.threshold_mapper)
        self.threshold_actor.InterpolateOff() 

        # prop picker for clicking on image
        self.picker = vtk.vtkPropPicker()
        

    def sliceChanged(self, slice_nr):
        self.slice_view_slider.setSliderPosition(slice_nr)
        self.mask_slice_mapper.SetSliceNumber(slice_nr)
        self.threshold_mapper.SetSliceNumber(slice_nr)
        # check what to update 
        if self.toolbar_auto_update.isChecked():
            self.model_view.GetRenderWindow().Render()
        else: 
            self.slice_view.GetRenderWindow().Render()
        

    def showEvent(self, event):
        self.slice_view.Enable()
        self.slice_view.EnableRenderOn()
        super(SegmentationModuleTab, self).showEvent(event)


    def hideEvent(self, event):
        self.slice_view.Disable()
        self.slice_view.EnableRenderOff()
        super(SegmentationModuleTab, self).hideEvent(event)  
    

    def loadVolumeSeg(self, volume_file, seg_file, pred_file, is_new_file=True):
        self.pred_file = pred_file
        if volume_file:
            # load image volume if it is new
            if is_new_file:
                self.image = self.slice_view.loadNrrd(volume_file)
                self.__loadImageData()
                self.brush_size = abs(self.image.GetSpacing()[0]*self.brush_size)  
                self.toolbar_edit.setEnabled(True)
                self.slice_view_slider.setRange(
                    self.slice_view.min_slice,
                    self.slice_view.max_slice
                )
                self.slice_view_slider.setSliderPosition(self.slice_view.slice)  
                self.slice_view_slider.setEnabled(True)
                self.slice_view_slider_label.setEnabled(True)
                

            # image exists -> load segmentation
            if seg_file:
                self.label_map, self.plaque_pending, self.lumen_pending = self.model_view.loadNrrd(seg_file, self.image)
                self.__loadLabelMapData()
                self.model_camera_pending = False
                self.model_view.renderer.AddActor(self.lumen_outline_actor3D)
                self.model_view.renderer.AddActor(self.plaque_outline_actor3D)
                if not self.editing_active:
                    self.slice_view.renderer.AddActor(self.lumen_outline_actor2D)
                    self.slice_view.renderer.AddActor(self.plaque_outline_actor2D)
            
            # image exists -> create empty segmentation
            else:
                self.plaque_pending = True
                self.lumen_pending = True
                self.model_camera_pending = True
                self.model_view.reset()
                self.label_map = vtk.vtkImageData()
                self.label_map.SetDimensions(self.image.GetDimensions())
                self.label_map.SetSpacing(self.image.GetSpacing())
                self.label_map.SetOrigin(self.image.GetOrigin())
                self.label_map_data = np.zeros(self.label_map.GetDimensions(), dtype=np.uint8)
                vtk_data_array = numpy_to_vtk(self.label_map_data.ravel(order='F'))
                self.label_map.GetPointData().SetScalars(vtk_data_array)
                self.masks_color_mapped.SetInputData(self.label_map)

            # initialize brush 
            self.toolbar_lumen.trigger()
            self.toolbar_brush2D.trigger() 
            self.thresholdChanged(self.threshold) 
            self.toolbar_auto_update.setChecked(True) 

            # draw scene
            self.slice_view.GetRenderWindow().Render()
            self.model_view.GetRenderWindow().Render()


        # no image -> reset
        else:
            self.plaque_pending = True
            self.lumen_pending = True
            self.model_camera_pending = True
            self.image = None
            self.image_data = None
            self.label_map = None
            self.label_map_data = None
            self.threshold_img = None
            self.deactivateEditing()
            self.edit_button.setEnabled(False)
            self.toolbar_edit.setEnabled(False)
            self.slice_view_slider.setEnabled(False)
            self.slice_view_slider_label.setEnabled(False)
            self.model_view.renderer.RemoveActor(self.lumen_outline_actor3D)
            self.slice_view.renderer.RemoveActor(self.lumen_outline_actor2D)
            self.model_view.renderer.RemoveActor(self.plaque_outline_actor3D)
            self.slice_view.renderer.RemoveActor(self.plaque_outline_actor2D)
            self.slice_view.reset()
            self.model_view.reset()


    def generateCNNSeg(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("New Segmentation: Initialize with CNN")
        if not self.pred_file:
            dlg.setText("Error: Could not generate segmentation prediction.")
            dlg.setStandardButtons(QMessageBox.Cancel)
            button = dlg.exec()
        else:
            dlg.setText("<p align='center'>Generate a segmentation prediction?<br>WARNING: Fully overwrites current mask!</p>")
            dlg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            button = dlg.exec()
            if button == QMessageBox.Ok:
                self.label_map, self.plaque_pending, self.lumen_pending = self.model_view.loadNrrd(self.pred_file)
                self.__loadLabelMapData()
                self.model_camera_pending = False
                self.model_view.renderer.AddActor(self.lumen_outline_actor3D)
                self.model_view.renderer.AddActor(self.plaque_outline_actor3D)
                if not self.editing_active:
                    self.slice_view.renderer.AddActor(self.lumen_outline_actor2D)
                    self.slice_view.renderer.AddActor(self.plaque_outline_actor2D)
                self.slice_view.GetRenderWindow().Render()
                self.model_view.GetRenderWindow().Render()
                self.data_modified.emit()

    
    def __loadLabelMapData(self):
        shape = self.label_map.GetDimensions()
        self.label_map_data = vtk_to_numpy(self.label_map.GetPointData().GetScalars())
        self.label_map_data = self.label_map_data.reshape(shape, order='F')
        self.masks_color_mapped.SetInputData(self.label_map)


    def __loadImageData(self):  # double underscore necessary?
        # image to display threshold
        shape = self.image.GetDimensions()
        position = self.image.GetOrigin()
        spacing = self.image.GetSpacing()
        
        self.threshold_img = vtk.vtkImageData()  
        self.threshold_img.SetDimensions(shape)
        self.threshold_img.SetOrigin(position)
        self.threshold_img.SetSpacing(spacing)

        self.image_data = vtk_to_numpy(self.image.GetPointData().GetScalars())
        self.image_data = self.image_data.reshape(shape, order='F')
        min, max = self.image_data.min(), self.image_data.max()
        self.threshold_slider.setMinimum(min)
        self.threshold_slider.setMaximum(max+1)
        self.threshold_color_mapped.SetInputData(self.threshold_img)


    def setUpCircle(self): 
        #set up circle to display around cursor and to display brush size  
        circle = vtk.vtkRegularPolygonSource() 
        circle.GeneratePolygonOff()
        circle.SetNumberOfSides(20)
        return circle
    
    def edit(self, on:bool):
        if on: 
            self.activateEditing()
        else: 
            self.deactivateEditing()

    def activateEditing(self):
        # enable all buttons needed for editing
        self.toolbar_lumen.setEnabled(True)
        self.toolbar_plaque.setEnabled(True)
        self.toolbar_brush2D.setEnabled(True)
        self.toolbar_brush3D.setEnabled(True)
        self.toolbar_auto_update.setEnabled(True)
        self.brush_size_slider.setEnabled(True)
        self.brush_slider_label.setEnabled(True) 
        self.threshold_slider.setEnabled(True)
        self.threshold_slider_label.setEnabled(True)
    
        # set observers for drawing 
        self.pickEvent = self.slice_view.interactor_style.AddObserver("MouseMoveEvent", self.pickPosition) 
        self.startLeft = self.slice_view.interactor_style.AddObserver("LeftButtonPressEvent", self.start_draw) 
        self.startRight = self.slice_view.interactor_style.AddObserver("RightButtonPressEvent", self.start_draw)

        # change scene
        self.slice_view.renderer.AddActor(self.circle_actor) 
        self.slice_view.renderer.RemoveActor(self.lumen_outline_actor2D)
        self.slice_view.renderer.RemoveActor(self.plaque_outline_actor2D)
        self.slice_view.renderer.AddActor(self.mask_slice_actor)
        self.slice_view.interactor_style.SetCurrentImageNumber(0)
        self.slice_view.GetRenderWindow().Render()
      
        
    def deactivateEditing(self):
        # disable all buttons and remove all actors for editing, enable editing again
        self.brush_size_slider.setEnabled(False)
        self.brush_slider_label.setEnabled(False)
        self.threshold_slider.setEnabled(False)
        self.threshold_slider_label.setEnabled(False)
        self.toolbar_lumen.setEnabled(False)
        self.toolbar_plaque.setEnabled(False)
        self.toolbar_brush2D.setEnabled(False)
        self.toolbar_brush3D.setEnabled(False)
        self.toolbar_auto_update.setEnabled(False)

        # remove observers for drawing, so that other observers work again 
        self.slice_view.interactor_style.RemoveObserver(self.startLeft) 
        self.slice_view.interactor_style.RemoveObserver(self.startRight) 
        self.slice_view.interactor_style.RemoveObserver(self.pickEvent)

        # change scene
        self.slice_view.renderer.RemoveActor(self.circle_actor)
        self.slice_view.renderer.AddActor(self.lumen_outline_actor2D)
        self.slice_view.renderer.AddActor(self.plaque_outline_actor2D)
        self.slice_view.renderer.RemoveActor(self.mask_slice_actor)
        self.slice_view.GetRenderWindow().Render()
        
    
    def brushSizeChanged(self, brush_size): 
        # change size of drawing on label map
        x_spacing = abs(self.image.GetSpacing()[0]) # can be negative
        self.brush_size = int(round(brush_size * x_spacing))
        self.circle.SetRadius(self.brush_size * x_spacing)

        # create circle mask
        axis = np.arange(-self.brush_size, self.brush_size+1, 1)
        if self.draw3D == False:  
            X, Y = np.meshgrid(axis, axis)  
            R = X**2 + Y**2
            R[R < self.brush_size**2] = 1
            R[R > 1] = 0
  
        elif self.draw3D == True:  
            z_scaling = abs(self.image.GetSpacing()[2]/self.image.GetSpacing()[0])
            self.brush_z = int(round(self.brush_size/z_scaling))
            axis_z = np.arange(-self.brush_z, self.brush_z+1, 1)
            z_axis = np.rint(axis_z*z_scaling)
            X, Y, Z = np.meshgrid(axis, axis, z_axis) 
            R = X**2 + Y**2 + Z**2
            R[R < self.brush_size**2]  = 1 
            R[R > 1] = 0 
        self.circle_mask = R.astype(np.bool_) 
        self.slice_view.GetRenderWindow().Render() 

    def set2DBrush(self, on:bool):
        if on:  # set up 2D brush 
            self.draw3D = False
            self.toolbar_brush3D.setChecked(False)

        else:  # set up 3D brush if button clicked second time 
            self.draw3D = True 
            self.toolbar_brush3D.setChecked(True)
        self.brushSizeChanged(abs(round(self.brush_size/self.image.GetSpacing()[0])))
        
    def set3DBrush(self, on:bool):
        
        if on:  # set up 3D brush 
            self.draw3D = True
            self.toolbar_brush2D.setChecked(False)
        else:
            self.draw3D = False  # set up 2D brush if button clicked second time
            self.toolbar_brush2D.setChecked(True)
        self.brushSizeChanged(abs(round(self.brush_size/self.image.GetSpacing()[0])))    


    def thresholdChanged(self,threshold):
        self.threshold = threshold
        self.threshold_slider_label.setText("Threshold: "+ str(self.threshold) + " (HU)")  # update slider label 

        # get copy of pixel values 
        threshold_img_data = np.copy(self.image_data)

        # define threshold mask 
        threshold_img_data[threshold_img_data<self.threshold] = 0
        threshold_img_data[threshold_img_data>self.threshold] = 1
        self.threshold_mask = threshold_img_data.astype(np.bool_)
        vtk_data_array = numpy_to_vtk(threshold_img_data.ravel(order='F'))
        self.threshold_img.GetPointData().SetScalars(vtk_data_array)
        
        # update scene 
        self.slice_view.GetRenderWindow().Render()

    # to show threshold only when slider moved
    def showThreshold(self):  
        self.slice_view.renderer.AddActor(self.threshold_actor)  
        self.slice_view.GetRenderWindow().Render()

    def hideThreshold(self): 
        self.slice_view.renderer.RemoveActor(self.threshold_actor)
        self.slice_view.GetRenderWindow().Render()  
        

    def setColorLumen(self, on:bool):
        if on:
            self.draw_value = 2
            self.circle_actor.GetProperty().SetColor(COLOR_LUMEN)  # set color of circle to lumen 
            self.toolbar_plaque.setChecked(False)
        else: 
            self.toolbar_plaque.trigger()

    def setColorPlaque(self, on:bool):
        if on:
            self.draw_value = 1
            self.circle_actor.GetProperty().SetColor(COLOR_PLAQUE)  # set color of circle to plaque 
            self.toolbar_lumen.setChecked(False)
        else: 
            self.toolbar_lumen.trigger()


    def pickPosition(self, obj, event):
        # pick current mouse position
        x,y = self.slice_view.GetEventPosition()  
        self.picker.Pick(x,y,self.slice_view.slice,self.slice_view.renderer) 
        position = self.picker.GetPickPosition()  # world coordinates 
        origin = self.image.GetOrigin()
        self.imgPos = ((position[0]-origin[0])/self.image.GetSpacing()[0], 
                       (position[1]-origin[1])/self.image.GetSpacing()[1], 
                       self.slice_view.slice)  # convert into image position

        self.circle.SetCenter(position[0],
                              position[1],
                              self.image.GetOrigin()[2]-self.image.GetExtent()[2])  # move circle if mouse moved 
        self.slice_view.GetRenderWindow().Render()


    def start_draw(self, obj, event):
        if event == "RightButtonPressEvent": # check if left (-> brush) or right (-> eraser) mouse button pressed
            self.eraser = True 

        # draw first point at position clicked on
        self.draw(obj,event)

        # check if pipeline needs updates
        if self.plaque_pending and self.draw_value == 1.0:
            self.model_view.renderer.AddActor(self.model_view.actor_plaque)
            self.model_view.renderer.ResetCamera()
            self.plaque_pending = False
            self.slice_view.GetRenderWindow().Render()
        elif self.lumen_pending and self.draw_value == 2.0:
            self.model_view.renderer.AddActor(self.model_view.actor_lumen)
            self.model_view.renderer.ResetCamera()
            self.lumen_pending = False
            self.slice_view.GetRenderWindow().Render()

        # draw as long as left mouse button pressed down 
        self.drawEvent = self.slice_view.interactor_style.AddObserver("MouseMoveEvent", self.draw) 
        if self.eraser: 
            self.endEvent = self.slice_view.interactor_style.AddObserver("RightButtonReleaseEvent", self.end_draw)
        else:
            self.endEvent = self.slice_view.interactor_style.AddObserver("LeftButtonReleaseEvent", self.end_draw) 
        self.data_modified.emit()
    
    def draw(self, obj, event):  
        # get discrete click position on label map
        x = int(round(self.imgPos[0]))
        y = int(round(self.imgPos[1]))
        z = int(self.slice_view.slice)

        # test if out of bounds
        if x < 0 or y < 0 or z < 0:
            return

        s = self.brush_size
        x0 = max(x-s, 0)
        x1 = min(x+s+1, self.label_map_data.shape[0])
        y0 = max(y-s, 0)
        y1 = min(y+s+1, self.label_map_data.shape[1])
        if self.draw3D == False:
            # draw circle  
            mask = np.copy(self.circle_mask[x0-x+s:x1-x+s, y0-y+s:y1-y+s]) # crop circle mask at borders

            # check if eraser or brush 
            if self.eraser: 
                # erase only current draw value 
                mask[self.label_map_data[x0:x1,y0:y1,z] != self.draw_value] = False   
                self.label_map_data[x0:x1,y0:y1,z][mask] = 0
            else:
                # draw only if HU above threshold 
                threshold = self.threshold_mask[x0:x1,y0:y1,z]  
                mask = threshold & mask
                self.label_map_data[x0:x1,y0:y1,z][mask] = self.draw_value


        else: 
            # draw sphere
            s_z = self.brush_z
            z0 = max(z-s_z, 0)
            z1 = min(z+s_z+1, self.label_map_data.shape[2])
            mask = np.copy(self.circle_mask[x0-x+s:x1-x+s, y0-y+s:y1-y+s, z0-z+s_z:z1-z+s_z]) # crop sphere mask at borders

            # check if eraser or brush 
            if self.eraser: 
                 # erase only current draw value
                mask[self.label_map_data[x0:x1,y0:y1,z0:z1] != self.draw_value] = False  
                self.label_map_data[x0:x1,y0:y1,z0:z1][mask] = 0
            else:
                # draw only if HU above threshold
                threshold = self.threshold_mask[x0:x1,y0:y1,z0:z1]
                mask = threshold & mask
                self.label_map_data[x0:x1,y0:y1,z0:z1][mask] = self.draw_value  

        # update the label map (shallow copies make this efficient)   
        vtk_data_array = numpy_to_vtk(self.label_map_data.ravel(order='F'))
        self.label_map.GetPointData().SetScalars(vtk_data_array)
        self.slice_view.GetRenderWindow().Render()
        
        
    def end_draw(self, obj, event):
        self.slice_view.interactor_style.RemoveObserver(self.drawEvent)  
        self.slice_view.interactor_style.RemoveObserver(self.endEvent) 

        if self.toolbar_auto_update.isChecked():  # update if auto-update is checked
            self.model_view.padding.SetInputData(self.label_map)
            self.model_view.GetRenderWindow().Render()

        if self.eraser:
            self.eraser = False 
        

    def saveChanges(self, path_seg, path_lumen, path_plaque):
        # catch if one side has something to save, other side not
        if self.label_map is None:
            return

        x_dim, y_dim, z_dim = self.label_map.GetDimensions()
        if x_dim == 0 or y_dim == 0 or z_dim == 0:
            return

        # save segmentation nrrd
        sx, sy, sz = self.label_map.GetSpacing()
        ox, oy, oz = self.label_map.GetOrigin()
        header = OrderedDict()
        header['type'] = 'unsigned char'
        header['dimension'] = 3
        header['space'] = 'left-posterior-superior'
        header['sizes'] = '120 144 248' # fixed model size
        header['space directions'] = [[sx, 0, 0], [0, sy, 0], [0, 0, sz]]
        header['kinds'] = ['domain', 'domain', 'domain']
        header['endian'] = 'little'
        header['encoding'] = 'gzip'
        header['space origin'] = [ox, oy, oz]
        header['Segment0_ID'] = 'Segment_1'
        header['Segment0_Name'] = 'plaque'
        header['Segment0_Color'] = str(241/255) + ' ' + str(214/255) + ' ' + str(145/255)
        header['Segment0_LabelValue'] = 1
        header['Segment0_Layer'] = 0
        header['Segment0_Extent'] = '0 119 0 143 0 247'
        header['Segment1_ID'] = 'Segment_2'
        header['Segment1_Name'] = 'lumen'
        header['Segment1_Color'] = str(216/255) + ' ' + str(101/255) + ' ' + str(79/255)
        header['Segment1_LabelValue'] = 2
        header['Segment1_Layer'] = 0
        header['Segment1_Extent'] = '0 119 0 143 0 247'
        segmentation = vtk_to_numpy(self.label_map.GetPointData().GetScalars())
        segmentation = segmentation.reshape(x_dim, y_dim, z_dim, order='F')
        nrrd.write(path_seg, segmentation, header)

        # save models
        writer = vtk.vtkSTLWriter()
        lumen = self.model_view.smoother_lumen.GetOutput()
        if lumen.GetNumberOfPoints() > 0:
            writer.SetFileName(path_lumen)
            writer.SetInputData(lumen)
            writer.Write()
        plaque = self.model_view.smoother_plaque.GetOutput()
        if plaque.GetNumberOfPoints() > 0:
            writer.SetFileName(path_plaque)
            writer.SetInputData(plaque)
            writer.Write()


    def close(self):
        self.slice_view.Finalize()
        self.model_view.Finalize()



class SegmentationModule(QTabWidget):
    """
    Module for segmenting the left/right carotid.
    """
    new_segmentation = pyqtSignal()
    new_models = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.patient_dict = None

        self.segmentation_module_left = SegmentationModuleTab()
        self.segmentation_module_right = SegmentationModuleTab()

        self.segmentation_module_left.data_modified.connect(self.dataModifiedLeft)
        self.segmentation_module_right.data_modified.connect(self.dataModifiedRight)

        self.addTab(self.segmentation_module_right, "Right")
        self.addTab(self.segmentation_module_left, "Left")


    def loadPatient(self, patient_dict):
        self.patient_dict = patient_dict
        self.segmentation_module_right.loadVolumeSeg(
            patient_dict['volume_right'], patient_dict['seg_right'], patient_dict['seg_right_pred'])
        self.segmentation_module_left.loadVolumeSeg(
            patient_dict['volume_left'], patient_dict['seg_left'], patient_dict['seg_left_pred'])


    def dataModifiedRight(self):
        self.setTabText(0, "Right " + SYM_UNSAVED_CHANGES)


    def dataModifiedLeft(self):
        self.setTabText(1, "Left " + SYM_UNSAVED_CHANGES)


    def discard(self):
        self.segmentation_module_right.loadVolumeSeg(
            self.patient_dict['volume_right'], self.patient_dict['seg_right'], self.patient_dict['seg_right_pred'], False)
        self.segmentation_module_left.loadVolumeSeg(
            self.patient_dict['volume_left'], self.patient_dict['seg_left'], self.patient_dict['seg_left_pred'], False)
        self.setTabText(0, "Right")
        self.setTabText(1, "Left")


    def save(self):
        patient_ID = self.patient_dict['patient_ID']
        base_path  = self.patient_dict['base_path']

        path_seg = os.path.join(base_path, patient_ID + "_right.seg.nrrd")
        path_lumen = os.path.join(base_path, "models", patient_ID + "_right_lumen.stl")
        path_plaque = os.path.join(base_path, "models", patient_ID + "_right_plaque.stl")
        self.segmentation_module_right.saveChanges(path_seg, path_lumen, path_plaque)
        
        path_seg = os.path.join(base_path, patient_ID + "_left.seg.nrrd")
        path_lumen = os.path.join(base_path, "models", patient_ID + "_left_lumen.stl")
        path_plaque = os.path.join(base_path, "models", patient_ID + "_left_plaque.stl")
        self.segmentation_module_left.saveChanges(path_seg, path_lumen, path_plaque)

        self.setTabText(0, "Right")
        self.setTabText(1, "Left")

        self.new_segmentation.emit()
        self.new_models.emit()


    def close(self):
        self.segmentation_module_right.close()
        self.segmentation_module_left.close()
