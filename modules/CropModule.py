from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSlider

from modules.ImageSliceInteractor import ImageSliceInteractor

class CropModule(QWidget):
    """
    Module for cropping the left/right carotid from a full CTA volume.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.slice_view_slider = QSlider(Qt.Horizontal)
        self.slice_view = ImageSliceInteractor(self.slice_view_slider, self)

        self.slice_view_layout = QVBoxLayout(self)
        self.slice_view_layout.addWidget(self.slice_view_slider)
        self.slice_view_layout.addWidget(self.slice_view)

        self.slice_view.Initialize()
        self.slice_view.Start()


    def showEvent(self, event):
        self.slice_view.Enable()
        self.slice_view.EnableRenderOn()
        super(CropModule, self).showEvent(event)


    def hideEvent(self, event):
        self.slice_view.Disable()
        self.slice_view.EnableRenderOff()
        super(CropModule, self).hideEvent(event)
    

    def load_patient(self, patient_dict):
        load_path = patient_dict['volume_left']
        if load_path:
            self.slice_view.loadNrrd(load_path)
        else:
            self.slice_view.reset()


    def close(self):
        self.slice_view.Finalize()