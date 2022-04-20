import vtk
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import  (
    QWidget, QVBoxLayout, QHBoxLayout, QSlider, QTabWidget,
    QPushButton, QMessageBox
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
        self.pred_file = False
        
        # on-screen objects
        self.CNN_button = QPushButton("New Segmentation: Initialize with CNN")
        self.slice_view = ImageSliceInteractor(self)
        self.slice_view.renderer.GetActiveCamera().SetViewUp(0, 1, 0)
        self.slice_view_slider = QSlider(Qt.Horizontal)
        self.model_view = IsosurfaceInteractor(self)
        self.model_view.renderer.GetActiveCamera().SetViewUp(0, 1, 0)

        # add everything to a layout
        self.slice_view_layout = QVBoxLayout()
        self.slice_view_layout.addWidget(self.CNN_button)
        self.slice_view_layout.addWidget(self.slice_view_slider)
        self.slice_view_layout.addWidget(self.slice_view)

        self.top_layout = QHBoxLayout(self)
        self.top_layout.addLayout(self.slice_view_layout)
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


    def close(self):
        self.slice_view.Finalize()
        self.model_view.Finalize()



class SegmentationModule(QTabWidget):
    """
    Module for segmenting the left/right carotid.
    """
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
        print("Save method called on Segmentation Module!")
        print("Dummy")


    def close(self):
        self.segmentation_module_right.close()
        self.segmentation_module_left.close()
