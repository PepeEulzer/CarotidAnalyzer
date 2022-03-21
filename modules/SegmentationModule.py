from os import path
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import  (
    QWidget, QVBoxLayout, QHBoxLayout, QSlider, QTabWidget,
    QPushButton
)
from vmtk import vmtkscripts

from modules.Interactors import ImageSliceInteractor, IsosurfaceInteractor

class SegmentationModuleTab(QWidget):
    """
    Tab view of a right OR left side carotid for segmentation.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.slice_view_slider = QSlider(Qt.Horizontal)
        self.slice_view = ImageSliceInteractor(self.slice_view_slider, self)
        self.model_view = IsosurfaceInteractor(self)


        self.slice_view_layout = QVBoxLayout()
        self.slice_view_layout.addWidget(self.slice_view_slider)
        self.slice_view_layout.addWidget(self.slice_view)

        self.top_layout = QHBoxLayout(self)
        self.top_layout.addLayout(self.slice_view_layout)
        self.top_layout.addWidget(self.model_view)

        self.slice_view.Initialize()
        self.slice_view.Start()
        self.model_view.Initialize()
        self.model_view.Start()


    def showEvent(self, event):
        self.slice_view.Enable()
        self.slice_view.EnableRenderOn()
        super(SegmentationModuleTab, self).showEvent(event)


    def hideEvent(self, event):
        self.slice_view.Disable()
        self.slice_view.EnableRenderOff()
        super(SegmentationModuleTab, self).hideEvent(event)
    

    def load_volume_seg(self, volume_file, seg_file):
        if volume_file:
            self.slice_view.loadNrrd(volume_file)
        else:
            self.slice_view.reset()
        
        if seg_file:
            self.model_view.loadNrrd(seg_file)
        else:
            self.model_view.reset()


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
        self.segmentation_module_left.load_volume_seg(
            patient_dict['volume_left'], patient_dict['seg_left'])
        self.segmentation_module_right.load_volume_seg(
            patient_dict['volume_right'], patient_dict['seg_right'])


    def close(self):
        self.segmentation_module_left.close()
        self.segmentation_module_right.close()
