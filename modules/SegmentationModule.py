import os
from collections import OrderedDict
from re import I

import nrrd
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import  (
    QWidget, QVBoxLayout, QHBoxLayout, QSlider, QTabWidget,
    QPushButton, QMessageBox, QGridLayout
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
        self.volume_file = False
        self.pred_file = False
        self.editing_active = False
        self.brush_size = 0
        
        # on-screen objects
        self.CNN_button = QPushButton("New Segmentation: Initialize with CNN")
        self.edit_button = QPushButton("Edit Segmentation")  # TO-DO: enable=false if no data loaded 
        self.brush_button = QPushButton("Brush") 
        self.eraser_button = QPushButton("Eraser")
        self.stop_editing_button = QPushButton("Stop Editing")
        self.lumen_button = QPushButton("Lumen")
        self.plaque_button = QPushButton("Plaque")
        self.brush_size_slider = QSlider(Qt.Horizontal)  # label?/somehow display brush size around cursor (maybe similar to 3D Slicer)
        self.brush_size_slider.setMinimum(0)
        self.brush_size_slider.setMaximum(10)
        self.brush_size_slider.setSingleStep(1)
        self.brush_size_slider.setValue(0)
        self.brush_button.setVisible(False)
        self.eraser_button.setVisible(False)
        self.stop_editing_button.setVisible(False)
        self.lumen_button.setVisible(False)
        self.plaque_button.setVisible(False)
        self.brush_size_slider.setVisible(False)
        self.slice_view = ImageSliceInteractor(self)
        self.slice_view.renderer.GetActiveCamera().SetViewUp(0, 1, 0)
        self.slice_view_slider = QSlider(Qt.Horizontal)
        self.model_view = IsosurfaceInteractor(self)
        self.model_view.renderer.GetActiveCamera().SetViewUp(0, 1, 0)

        # add everything to a layout
        self.slice_view_layout = QVBoxLayout()
        self.slice_view_layout.addWidget(self.CNN_button)
        self.slice_view_layout.addWidget(self.edit_button)
        self.slice_view_layout.addWidget(self.slice_view_slider)
        self.slice_view_layout.addWidget(self.slice_view)

        #self.edit_buttons_layout = QVBoxLayout()
        self.edit_buttons_layout = QGridLayout()
        self.edit_buttons_layout.addWidget(self.brush_button, 0,0,1,2) 
        self.edit_buttons_layout.addWidget(self.lumen_button, 1,0)
        self.edit_buttons_layout.addWidget(self.plaque_button, 1,1)
        self.edit_buttons_layout.addWidget(self.brush_size_slider, 2,0,1,2)
        self.edit_buttons_layout.addWidget(self.eraser_button, 3,0,1,2)
        self.edit_buttons_layout.addWidget(self.stop_editing_button, 4,0,1,2)

        self.top_layout = QHBoxLayout(self)
        self.top_layout.addLayout(self.slice_view_layout)
        self.top_layout.addLayout(self.edit_buttons_layout)
        self.top_layout.addWidget(self.model_view)
        

        # shared vtk objects
        self.lumen_outline_actor3D, self.lumen_outline_actor2D = self.__createOutlineActors(
            self.model_view.smoother_lumen.GetOutputPort(), COLOR_LUMEN_DARK, COLOR_LUMEN)
        self.plaque_outline_actor3D, self.plaque_outline_actor2D = self.__createOutlineActors(
            self.model_view.smoother_plaque.GetOutputPort(), COLOR_PLAQUE_DARK, COLOR_PLAQUE)

        # connect signals/slots
        self.CNN_button.pressed.connect(self.generateCNNSeg)
        self.slice_view.slice_changed[int].connect(self.sliceChanged)
        self.slice_view_slider.valueChanged[int].connect(self.slice_view.setSlice)
        self.slice_view_slider.valueChanged[int].connect(self.setSliceEditor)
        self.edit_button.pressed.connect(self.activateEditing)
        self.brush_button.pressed.connect(self.drawMode)  # put elsewhere? 
        self.eraser_button.pressed.connect(self.erase)
        self.brush_size_slider.valueChanged[int].connect(self.brushSizeChanged)
        self.stop_editing_button.pressed.connect(self.inactivateEditing)

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


    def sliceChanged(self, slice_nr):
        self.slice_view_slider.setSliderPosition(slice_nr)
        self.model_view.GetRenderWindow().Render()
        

    def showEvent(self, event):
        self.slice_view.Enable()
        self.slice_view.EnableRenderOn()
        super(SegmentationModuleTab, self).showEvent(event)


    def hideEvent(self, event):
        self.slice_view.Disable()
        self.slice_view.EnableRenderOff()
        super(SegmentationModuleTab, self).hideEvent(event)  
    

    def loadVolumeSeg(self, volume_file, seg_file, pred_file):
        self.volume_file = volume_file
        self.pred_file = pred_file
        if volume_file:
            self.slice_view.loadNrrd(volume_file, False)
            self.slice_view_slider.setRange(
                self.slice_view.min_slice,
                self.slice_view.max_slice
            )
            self.slice_view_slider.setSliderPosition(self.slice_view.slice)
        else:
            self.slice_view.reset()
        
        if seg_file:
            self.model_view.loadNrrd(seg_file)
            self.model_view.renderer.AddActor(self.lumen_outline_actor3D)
            self.slice_view.renderer.AddActor(self.lumen_outline_actor2D)
            self.model_view.renderer.AddActor(self.plaque_outline_actor3D)
            self.slice_view.renderer.AddActor(self.plaque_outline_actor2D)
        else:
            self.model_view.reset()
            self.model_view.renderer.RemoveActor(self.lumen_outline_actor3D)
            self.slice_view.renderer.RemoveActor(self.lumen_outline_actor2D)
            self.model_view.renderer.RemoveActor(self.plaque_outline_actor3D)
            self.slice_view.renderer.RemoveActor(self.plaque_outline_actor2D)


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
                self.model_view.loadNrrd(self.pred_file)
                self.model_view.renderer.AddActor(self.lumen_outline_actor3D)
                self.slice_view.renderer.AddActor(self.lumen_outline_actor2D)
                self.model_view.renderer.AddActor(self.plaque_outline_actor3D)
                self.slice_view.renderer.AddActor(self.plaque_outline_actor2D)
                self.data_modified.emit()


    def activateEditing(self):
        ### catch: no patient data loaded/hide button as long as no patient data loaded
        
        self.editing_active = True 
        self.brush_button.setVisible(True)
        self.eraser_button.setVisible(True)
        self.stop_editing_button.setVisible(True)
        self.edit_button.setEnabled(False)

        # define lookup-table for label map
        self.lookuptable = vtk.vtkLookupTable()
        self.lookuptable.SetNumberOfTableValues(3)
        self.lookuptable.SetTableValue(0, 0, 0, 0, 0)  # set color of backround (idx 0) to black with transparency 0
        alpha = (0.75,)
        self.lumen_rgba = COLOR_LUMEN + alpha
        self.plaque_rgba = COLOR_PLAQUE + alpha  # plaque also red ?  -  color and transperancy changes when clicked left and mouse wheel (only when in editing mode, does not work when clicked on brush)
        self.lookuptable.SetTableValue(1, self.plaque_rgba)
        self.lookuptable.SetTableValue(2, self.lumen_rgba)
        self.lookuptable.Build()  

        self.masks_color_mapped = vtk.vtkImageMapToColors()  
        self.masks_color_mapped.SetInputData(self.model_view.label_map)  
        self.masks_color_mapped.SetLookupTable(self.lookuptable)
        
        self.mask_slice_mapper = vtk.vtkOpenGLImageSliceMapper()  
        self.mask_slice_mapper.SetInputConnection(self.masks_color_mapped.GetOutputPort())
        self.mask_slice_mapper.SliceAtFocalPointOff() 
        self.mask_slice_mapper.SetNumberOfThreads(1)
        self.mask_slice_mapper.SetSliceNumber(self.slice_view.slice)
        
        self.mask_slice_actor = vtk.vtkImageActor()
        self.mask_slice_actor.SetMapper(self.mask_slice_mapper)
        self.mask_slice_actor.SetPosition(self.slice_view.image_actor.GetPosition())
        
        # define canvas to draw changes of segmentation on 
        self.canvas = self.setUpCanvas()
        self.canvas_actor = vtk.vtkImageActor()
        self.canvas_actor.GetMapper().SetInputConnection(self.canvas.GetOutputPort())
        img_pos = self.slice_view.image_actor.GetPosition()
        self.canvas_actor.SetPosition(img_pos)  

        self.slice_view.renderer.RemoveActor(self.lumen_outline_actor2D)
        self.slice_view.renderer.RemoveActor(self.plaque_outline_actor2D)
        self.slice_view.renderer.AddActor(self.mask_slice_actor)
        self.slice_view.renderer.AddActor(self.canvas_actor)
        ## is canvas displayed at right position (on top of image slice -> SetPosition of canvas actor)?
        self.slice_view.GetRenderWindow().Render()
        self.data_modified.emit() 
      
        
    def inactivateEditing(self):
        # show message, that changes will be lost 
        self.editing_active = False
        self.brush_button.setVisible(False)
        self.eraser_button.setVisible(False)
        self.stop_editing_button.setVisible(False)
        self.lumen_button.setVisible(False)
        self.plaque_button.setVisible(False)
        self.brush_size_slider.setVisible(False)
        self.edit_button.setEnabled(True)
        self.slice_view.renderer.AddActor(self.lumen_outline_actor2D)
        self.slice_view.renderer.AddActor(self.plaque_outline_actor2D)
        self.slice_view.renderer.RemoveActor(self.mask_slice_actor)
        self.slice_view.renderer.RemoveActor(self.canvas_actor)
        self.slice_view.GetRenderWindow().Render()
    
    def setUpCanvas(self):
        # if in activateEditing -> crash 
        colors = vtk.vtkNamedColors()
        canvas = vtk.vtkImageCanvasSource2D()
        lm_extent = self.model_view.label_map.GetExtent()
        canvas.SetExtent(lm_extent[0], lm_extent[1], lm_extent[2], lm_extent[3], self.slice_view.slice, self.slice_view.slice)  # instead of slice 0,0
        canvas.SetDrawColor(colors.GetColor4ub('DarkCyan'))
        canvas.FillTriangle(10,10,25,10,25,25)  # test if canvas visible, no triangle there 
        canvas.Update()
        return canvas

    
    def setSliceEditor(self):
        # connect mask slices to current image slice
        if not self.editing_active:
            return   
        self.mask_slice_mapper.SetSliceNumber(self.slice_view.slice)
        #self.canvas = self.setUpCanvas()
        self.slice_view.GetRenderWindow().Render()


    def brushSizeChanged(self, brush_size):
        # to-do: display current size
        self.brush_size = brush_size  # connect to draw
        #print(self.brush_size)

    # possible to hand over color rather than using two seperate methods?
    def setColorLumen(self):
        self.canvas.SetDrawColor(self.lumen_rgba)
    def setColorPlaque(self):
        self.canvas.SetDrawColor(self.plaque_rgba)

    # merge methods for drawing? 
    def drawMode(self):  # merge with draw? (put observer in ActivateEditing  and only use draw)
        self.slice_view.interactor_style.AddObserver("LeftButtonPressEvent", self.draw)  # remove at some point?
        self.lumen_button.setVisible(True)  # set false if exiting drawing/enable only if "draw" clicked 
        self.plaque_button.setVisible(True)
        self.brush_size_slider.setVisible(True)
        self.lumen_button.pressed.connect(self.setColorLumen)  # to-do: show that selected if clicked
        self.plaque_button.pressed.connect(self.setColorPlaque)

    def start_draw(self, obj, event):
        self.draw(obj, event)  # draw first point at posttion clicked on 
        # draw as long as left mouse button pressed down 
        self.slice_view.interactor_style.AddObserver("MouseMoveEvent", self.draw)  
        self.slice_view.interactor_style.AddObserver("LeftButtonReleaseEvent", self.end_draw)

    def draw(self, object, event):  
        # get position in image where clicked on -> draw via canvas 
        # mouse position instead of event position?
        x,y = self.slice_view.GetEventPosition()  
        picker = vtk.vtkCellPicker()  # other picker?
        picker.Pick(x,y,self.slice_view.slice,self.slice_view.renderer)  
        imgPos = picker.GetPointIJK()
        
        # catch: not selected if lumen/plaque (set draw color?)

        # nothing drawn on the image but position printed in prompt (if print not commented out)
        ## draw point at picked position 
        self.canvas.DrawPoint(imgPos[0],imgPos[1])
        ## draw box at picked position 
        #self.canvas.FillBox(imgPos[0],imgPos[0]+self.brush_size,imgPos[1],imgPos[1]+self.brush_size)  # (probably) does not draw on actual pixel position - better use drawpoint? any alternative method ?
        ## draw circle around picked position 
        #self.canvas.DrawCircle(imgPos[0], imgPos[1], self.brush_size)
        ## and fill area around circle with color
        #self.canvas.FillPixel(imgPos[0],imgPos[1])  # leads to Generic Warning: In ..\Imaging\Sources\vtkImageCanvasSource2D.cxx, line 1219 Fill: Cannot handle draw color same as fill color
       
        # print("Click Position", self.lastImgPoint) # worldPoint)

        #self.data_modified.emit()  
        self.slice_view.GetRenderWindow().Render()
    
    def end_draw(self, obj, event):
        id = vtk.vtkCommand.MouseMoveEvent
        self.slice_view.interactor_style.RemoveObservers(id)

    def erase(self):
        return 

    def saveChanges(self, path_seg, path_lumen, path_plaque):
        # catch if one side has something to save, other side not
        x_dim, y_dim, z_dim = self.model_view.label_map.GetDimensions()
        if x_dim == 0 or y_dim == 0 or z_dim == 0:
            return

        # save segmentation nrrd
        header_img = nrrd.read_header(self.volume_file)
        header = OrderedDict()
        header['type'] = 'unsigned char'
        header['dimension'] = 3
        header['space'] = 'left-posterior-superior'
        header['sizes'] = '120 144 248' # fixed model size
        header['space directions'] = header_img['space directions']
        header['kinds'] = ['domain', 'domain', 'domain']
        header['endian'] = 'little'
        header['encoding'] = 'gzip'
        header['space origin'] = header_img['space origin']
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
        segmentation = vtk_to_numpy(self.model_view.label_map.GetPointData().GetScalars())
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

        self.addTab(self.segmentation_module_right, "Right")
        self.addTab(self.segmentation_module_left, "Left")


    def loadPatient(self, patient_dict):
        self.patient_dict = patient_dict
        self.segmentation_module_right.loadVolumeSeg(
            patient_dict['volume_right'], patient_dict['seg_right'], patient_dict['seg_right_pred'])
        self.segmentation_module_left.loadVolumeSeg(
            patient_dict['volume_left'], patient_dict['seg_left'], patient_dict['seg_left_pred'])


    def discard(self):
        self.loadPatient(self.patient_dict)


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

        self.new_segmentation.emit()
        self.new_models.emit()


    def close(self):
        self.segmentation_module_right.close()
        self.segmentation_module_left.close()
