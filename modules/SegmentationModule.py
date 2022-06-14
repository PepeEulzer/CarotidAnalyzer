import os
from collections import OrderedDict
from re import I

import nrrd
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import  (
    QWidget, QVBoxLayout, QHBoxLayout, QSlider, QTabWidget,
    QPushButton, QMessageBox, QGridLayout, QLabel
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
        self.image = None
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
        #self.brush_size_slider.setTickPosition(QSlider.TicksBelow)
        self.brush_size_slider.setTickInterval(1)
        self.brush_slider_label = QLabel()
        self.brush_slider_label.setText("Brush Size")  # NOT SHOWING
        self.brush_button.setVisible(False)
        self.eraser_button.setVisible(False)
        self.stop_editing_button.setVisible(False)
        self.lumen_button.setVisible(False)
        self.plaque_button.setVisible(False)
        self.brush_size_slider.setVisible(False)
        self.brush_slider_label.setVisible(False)
        self.slice_view = ImageSliceInteractor(self)
        self.slice_view.renderer.GetActiveCamera().SetViewUp(0, 1, 0)
        self.slice_view_slider = QSlider(Qt.Horizontal)
        self.model_view = IsosurfaceInteractor(self)
        self.model_view.renderer.GetActiveCamera().SetViewUp(0, 1, 0)

        # add everything to a layout
        self.slice_view_layout = QVBoxLayout()
        self.slice_view_layout.addWidget(self.CNN_button)
        self.slice_view_layout.addWidget(self.edit_button)
        #self.slice_view_layout.addWidget(self.brush_slider_label)
        self.slice_view_layout.addWidget(self.slice_view_slider)
        self.slice_view_layout.addWidget(self.slice_view)

        #self.edit_buttons_layout = QVBoxLayout()
        self.edit_buttons_layout = QGridLayout()
        self.edit_buttons_layout.addWidget(self.brush_button, 0,0,1,2) 
        self.edit_buttons_layout.addWidget(self.lumen_button, 1,0)
        self.edit_buttons_layout.addWidget(self.plaque_button, 1,1)
        self.edit_buttons_layout.addWidget(self.brush_slider_label, 2,0,1,2)
        self.edit_buttons_layout.addWidget(self.brush_size_slider, 3,0,1,2)
        self.edit_buttons_layout.addWidget(self.eraser_button, 4,0,1,2)
        self.edit_buttons_layout.addWidget(self.stop_editing_button, 5,0,1,2)

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
        self.lumen_button.pressed.connect(self.setColorLumen)  # to-do: show that selected if clicked
        self.plaque_button.pressed.connect(self.setColorPlaque)

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
            self.image = self.slice_view.loadNrrd(volume_file, False)
            self.slice_view_slider.setRange(
                self.slice_view.min_slice,
                self.slice_view.max_slice
            )
            self.slice_view_slider.setSliderPosition(self.slice_view.slice)

        else:
            self.slice_view.reset()
            self.image = None
        
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
        ## would it be better to save initialized object when editing mode is acitvated once and then to reuse them when activated again?
        self.editing_active = True 
        self.brush_button.setVisible(True)
        self.eraser_button.setVisible(True)
        self.stop_editing_button.setVisible(True)
        self.edit_button.setEnabled(False)

        # define lookup-table for label map -> put label map into canvas
        self.lookuptable =vtk.vtkLookupTable() # vtk.vtkWindowLevelLookupTable()
        self.lookuptable.SetNumberOfTableValues(3)
        self.lookuptable.SetTableRange(0,2)
        self.lookuptable.SetTableValue(0, 0.0, 0.0, 0.0, 0.0)  # set color of backround (id 0) to black with transparency 0
        alpha = (0.75,)
        self.lumen_rgba = COLOR_LUMEN + alpha
        self.plaque_rgba = COLOR_PLAQUE + alpha  # plaque also red ?  -  color and transperancy changes when clicked left and mouse wheel (only when in editing mode, does not work when clicked on brush)
        self.lookuptable.SetTableValue(1, self.plaque_rgba)
        self.lookuptable.SetTableValue(2, self.lumen_rgba)
        self.lookuptable.Build() 
        
        self.masks_color_mapped = vtk.vtkImageMapToColors() 
        self.masks_color_mapped.SetLookupTable(self.lookuptable) 
        self.masks_color_mapped.PassAlphaToOutputOn()
        self.masks_color_mapped.SetInputData(self.model_view.label_map)  
        
        self.mask_slice_mapper = vtk.vtkOpenGLImageSliceMapper()  
        self.mask_slice_mapper.SetInputConnection(self.masks_color_mapped.GetOutputPort())
        self.mask_slice_mapper.SetSliceNumber(self.slice_view.slice)
        
        self.mask_slice_actor = vtk.vtkImageActor()
        self.mask_slice_actor.InterpolateOff()
        self.mask_slice_actor.SetMapper(self.mask_slice_mapper)
        self.mask_slice_actor.SetPosition(self.slice_view.image_actor.GetPosition())
       

        # define canvas to draw changes of segmentation on 
        self.canvas = self.setUpCanvas()
        self.canvas_mapper = vtk.vtkOpenGLImageSliceMapper()
        self.canvas_mapper.SetInputConnection(self.canvas.GetOutputPort())
        self.canvas_mapper.SetSliceNumber(self.slice_view.slice)

        self.canvas_actor = vtk.vtkImageActor()  # ad actor below 
        self.canvas_actor.InterpolateOff()
        self.canvas_actor.GetProperty().SetOpacity(0.3) # any other solution to implement without opycity?
        self.canvas_actor.SetPosition(self.image.GetOrigin()[0],self.image.GetOrigin()[1],self.image.GetOrigin()[2]-self.slice_view.slice)  # (self.image.GetOrigin())
        self.canvas_actor.SetMapper(self.canvas_mapper)

        # circle around mouse when drawing 
        self.circle = self.setUpCircle()
        self.circle.SetCenter(self.image.GetOrigin()[0], self.image.GetOrigin()[1], self.image.GetOrigin()[2]-self.image.GetExtent()[2])
        circle_mapper = vtk.vtkPolyDataMapper()
        circle_mapper.SetInputConnection(self.circle.GetOutputPort())
        self.circle_actor = vtk.vtkActor()
        self.circle_actor.SetMapper(circle_mapper)
        color = vtk.vtkNamedColors()
        self.circle_actor.GetProperty().SetColor(color.GetColor3d('LightCyan'))  # SetColor(241,214,145)  # color not working -> other setting of color needed/sth similar to setnumberofscalarvalues in canvas needed??
        #self.circle_actor.SetPosition(self.image.GetOrigin()[0], self.image.GetOrigin()[1], self.image.GetOrigin()[2]-self.image.GetExtent()[2])

        self.slice_view.renderer.RemoveActor(self.lumen_outline_actor2D)
        self.slice_view.renderer.RemoveActor(self.plaque_outline_actor2D)
        self.slice_view.renderer.AddActor(self.mask_slice_actor)
        self.slice_view.renderer.AddActor(self.canvas_actor)  
        self.slice_view.GetRenderWindow().Render()
        self.data_modified.emit() 
      
        
    def inactivateEditing(self):
        # show message, that changes will be lost (or store temporally)
        self.editing_active = False
        self.brush_button.setVisible(False)
        self.eraser_button.setVisible(False)
        self.stop_editing_button.setVisible(False)
        self.lumen_button.setVisible(False)
        self.plaque_button.setVisible(False)
        self.brush_size_slider.setVisible(False)
        self.brush_slider_label.setVisible(False)
        self.edit_button.setEnabled(True)
        self.slice_view.renderer.AddActor(self.lumen_outline_actor2D)
        self.slice_view.renderer.AddActor(self.plaque_outline_actor2D)
        self.slice_view.renderer.RemoveActor(self.mask_slice_actor)
        self.slice_view.renderer.RemoveActor(self.canvas_actor)
        self.slice_view.renderer.RemoveActor(self.circle_actor)
        self.slice_view.GetRenderWindow().Render()
    
    def setUpCanvas(self):
        # if in activateEditing -> crash 
        canvas = vtk.vtkImageCanvasSource2D()
        img_spacing = self.image.GetSpacing() 
        ###canvas.SetRatio(img_spacing) -> by setting the extent this seems to happen automatically 
        # set extent to scalar neccessary? (in vtkimagedata class)
        img_extent = self.image.GetExtent()
        canvas.SetExtent(vtk.vtkMath.Ceil(img_extent[0]*img_spacing[0]), vtk.vtkMath.Floor(img_extent[1]*img_spacing[0]), vtk.vtkMath.Ceil(img_extent[2]*img_spacing[1]),vtk.vtkMath.Ceil(img_extent[3]*img_spacing[1]), vtk.vtkMath.Ceil(img_extent[4]*img_spacing[2]), vtk.vtkMath.Floor(img_extent[5]*img_spacing[2]))  # , self.slice_view.slice, self.slice_view.slice)  # (img_extent) #   # instead of slice 0,0
        #canvas.SetExtent(img_extent[0], img_extent[1],img_extent[2], img_extent[3]+1, img_extent[4], img_extent[5])  #  with this canvas to big
        canvas.SetDefaultZ(self.slice_view.slice)
        canvas.SetNumberOfScalarComponents(3)  # what exacly does this do?! doc: Set the number of scalar components; in vtkImageData doc: Set/Get the number of scalar components for points. As with the SetScalarType method this is setting pipeline info.
        canvas.SetDrawColor(0,0,0)  
        return canvas

    def setUpCircle(self): 
        #set up circle to display around cursor and to display brush size  
        circle = vtk.vtkRegularPolygonSource()  # is there a way to not do it via approximation of circle? 
        circle.GeneratePolygonOff()
        circle.SetNumberOfSides(30)
        circle.SetRadius(self.brush_size)  #*self.image.GetSpacing()[0] ? 
        return circle

    def setSliceEditor(self, slice_nr):
        # connect mask slices to current image slice
        if not self.editing_active:
            return   
        self.mask_slice_mapper.SetSliceNumber(slice_nr)  
        self.canvas.SetDefaultZ(self.slice_view.slice)  # also connected to position of actor !
        self.canvas_mapper.SetSliceNumber(slice_nr)
        self.canvas_actor.SetPosition(self.image.GetOrigin()[0],self.image.GetOrigin()[1],self.image.GetOrigin()[2]-slice_nr)  # 2/3 value by tryout (best found yet) but not all slices have canvas yet/ not 100 working
        self.slice_view.GetRenderWindow().Render()

    def brushSizeChanged(self, brush_size):
        # change size of drawing on canvas 
        self.brush_size = brush_size  
        self.circle.SetRadius(brush_size)  # *image spacing?? 
        self.slice_view.GetRenderWindow().Render()
    def setColorLumen(self):  
        self.canvas.SetDrawColor(216,101,79) 
    def setColorPlaque(self):
        self.canvas.SetDrawColor(241,214,145)
    
    def drawMode(self):  
        self.slice_view.interactor_style.AddObserver("MouseMoveEvent", self.pickPosition)
        self.slice_view.interactor_style.AddObserver("LeftButtonPressEvent", self.start_draw)  # remove at some point?
        #self.slice_view.interactor_style.AddObserver("MouseMoveEvent", self.circleCursor)
        #self.slice_view.renderer.AddActor(self.circle_actor)  # correct position? 
        self.lumen_button.setVisible(True)  # set false if exiting drawing/enable only if "draw" clicked 
        self.plaque_button.setVisible(True)
        self.brush_size_slider.setVisible(True)
        #self.brush_slider_label.setVisible(True)  # when label included positioning not great (possible to fix postion?)
        self.slice_view.renderer.AddActor(self.circle_actor)  # dissappers when sth is drawn on canvas 
        #self.circle.SetCenter()
        self.slice_view.GetRenderWindow().Render()
    

    def pickPosition(self, obj, event):
        x,y = self.slice_view.GetEventPosition()  
        picker = vtk.vtkPropPicker()
        picker.Pick(x,y,self.slice_view.slice,self.slice_view.renderer) 
        position = picker.GetPickPosition()  # world coordinates # (0.0,0.0,0.0) if pick3DPoint
        origin = self.image.GetOrigin()
        self.imgPos = ((position[0]-origin[0]), (position[1]-origin[1]), (position[2]+origin[2]))  # convert into image position
        ##imgPos = ((pos[0]-origin[0])/self.image.GetSpacing()[0], (pos[1]-origin[1])/self.image.GetSpacing()[1], (pos[2]+origin[2])/self.image.GetSpacing()[2])  -> needed when canvas.SetRatio() set 
        self.circle.SetCenter(position[0],position[1],self.image.GetOrigin()[2]-self.image.GetExtent()[2])  # move circle if mouse moved 
        #self.circle_actor.SetPosition(position[0], position[1], self.image.GetOrigin()[2]-self.image.GetExtent()[2])
        self.slice_view.GetRenderWindow().Render()

    def start_draw(self, obj, event):
        self.draw(obj, event)  # draw first point at posttion clicked on 
        # draw as long as left mouse button pressed down 
        self.slice_view.interactor_style.AddObserver("MouseMoveEvent", self.draw)  
        self.slice_view.interactor_style.AddObserver("LeftButtonReleaseEvent", self.end_draw)
    
    def draw(self, object, event): 
        #draw point
        #self.canvas.DrawPoint(vtk.vtkMath.Round(imgPos[0]),vtk.vtkMath.Roundround(imgPos[1]))
        ## draw box at picked position 
        #self.canvas.FillBox(imgPos[0],imgPos[0]+self.brush_size,imgPos[1],imgPos[1]+self.brush_size)  # (probably) does not draw on actual pixel position - better use drawpoint? any alternative method ?
        #draw circle around picked position 
        self.canvas.DrawCircle(vtk.vtkMath.Round(self.imgPos[0]), vtk.vtkMath.Round(self.imgPos[1]), self.brush_size)  # for bigger circles not everything filled out 
        ## and fill area around circle with color
        #self.canvas.FillPixel(imgPos[0],imgPos[1])  # leads to Generic Warning: In ..\Imaging\Sources\vtkImageCanvasSource2D.cxx, line 1219 Fill: Cannot handle draw color same as fill color
        self.slice_view.GetRenderWindow().Render()
    
    def end_draw(self, obj, event):
        id = vtk.vtkCommand.MouseMoveEvent
        self.slice_view.interactor_style.RemoveObservers(id)  # is it possible to just remove one of the observers?
        self.slice_view.interactor_style.AddObserver("MouseMoveEvent", self.pickPosition)

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
