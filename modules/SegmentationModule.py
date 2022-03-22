import vtk
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import  (
    QWidget, QVBoxLayout, QHBoxLayout, QSlider, QTabWidget,
    QPushButton
)

from modules.Interactors import ImageSliceInteractor, IsosurfaceInteractor

COLOR_LUMEN_DARK = (55/255, 22/255, 15/255)
COLOR_PLAQUE_DARK = (81/255, 69/255, 40/255)

COLOR_LUMEN = (216/255, 101/255, 79/255)
COLOR_PLAQUE = (241/255, 214/255, 145/255)

class SegmentationModuleTab(QWidget):
    """
    Tab view of a right OR left side carotid for segmentation.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # on-screen objects
        self.slice_view = ImageSliceInteractor(self)
        self.slice_view_slider = QSlider(Qt.Horizontal)
        self.model_view = IsosurfaceInteractor(self)

        # add everything to a layout
        self.slice_view_layout = QVBoxLayout()
        self.slice_view_layout.addWidget(self.slice_view_slider)
        self.slice_view_layout.addWidget(self.slice_view)

        self.top_layout = QHBoxLayout(self)
        self.top_layout.addLayout(self.slice_view_layout)
        self.top_layout.addWidget(self.model_view)

        # shared vtk objects
        self.lumen_outline_actor = self.__createOutlineActor(self.model_view.smoother_lumen.GetOutputPort(), COLOR_LUMEN)
        self.plaque_outline_actor = self.__createOutlineActor(self.model_view.smoother_plaque.GetOutputPort(), COLOR_PLAQUE)

        # connect signals/slots
        self.slice_view.slice_changed[int].connect(self.sliceChanged)
        self.slice_view_slider.valueChanged[int].connect(self.slice_view.setSlice)

        # initialize VTK
        self.slice_view.Initialize()
        self.slice_view.Start()
        self.model_view.Initialize()
        self.model_view.Start()

    def __createOutlineActor(self, output_port, color):
        cutter = vtk.vtkCutter()
        cutter.SetInputConnection(output_port)
        cutter.SetCutFunction(self.slice_view.image_mapper.GetSlicePlane())
        mapper = vtk.vtkPolyDataMapper()
        mapper.ScalarVisibilityOff()
        mapper.SetInputConnection(cutter.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetLineWidth(4)
        actor.GetProperty().RenderLinesAsTubesOn()
        return actor


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
    

    def loadVolumeSeg(self, volume_file, seg_file):
        if volume_file:
            self.slice_view.loadNrrd(volume_file)
            self.slice_view_slider.setRange(
                self.slice_view.min_slice,
                self.slice_view.max_slice
            )
            self.slice_view_slider.setSliderPosition(self.slice_view.slice)
        else:
            self.slice_view.reset()
        
        if seg_file:
            self.model_view.loadNrrd(seg_file)
            self.model_view.renderer.AddActor(self.lumen_outline_actor)
            self.slice_view.renderer.AddActor(self.lumen_outline_actor)
            self.model_view.renderer.AddActor(self.plaque_outline_actor)
            self.slice_view.renderer.AddActor(self.plaque_outline_actor)
        else:
            self.model_view.reset()
            self.model_view.renderer.RemoveActor(self.lumen_outline_actor)
            self.slice_view.renderer.RemoveActor(self.lumen_outline_actor)
            self.model_view.renderer.RemoveActor(self.plaque_outline_actor)
            self.slice_view.renderer.RemoveActor(self.plaque_outline_actor)


    def close(self):
        self.slice_view.Finalize()
        self.model_view.Finalize()



class SegmentationModule(QTabWidget):
    """
    Module for segmenting the left/right carotid.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.segmentation_module_left = SegmentationModuleTab()
        self.segmentation_module_right = SegmentationModuleTab()

        self.addTab(self.segmentation_module_left, "Left")
        self.addTab(self.segmentation_module_right, "Right")


    def load_patient(self, patient_dict):
        self.segmentation_module_left.loadVolumeSeg(
            patient_dict['volume_left'], patient_dict['seg_left'])
        self.segmentation_module_right.loadVolumeSeg(
            patient_dict['volume_right'], patient_dict['seg_right'])


    def close(self):
        self.segmentation_module_left.close()
        self.segmentation_module_right.close()
