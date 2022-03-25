from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QPushButton
from vmtk import vmtkscripts

from modules.Interactors import ImageSliceInteractor, VolumeRenderingInteractor

class CropModule(QWidget):
    """
    Module for cropping the left/right carotid from a full CTA volume.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.slice_view = ImageSliceInteractor(self)
        self.volume_view = VolumeRenderingInteractor(self)
        self.slice_view_slider = QSlider(Qt.Horizontal)
        self.button_set_left = QPushButton("Set Left Volume")
        self.button_set_right = QPushButton("Set Right Volume")

        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.button_set_right)
        self.button_layout.addWidget(self.button_set_left)
        self.slice_view_layout = QVBoxLayout()
        self.slice_view_layout.addLayout(self.button_layout)
        self.slice_view_layout.addWidget(self.slice_view_slider)
        self.slice_view_layout.addWidget(self.slice_view)
        self.top_layout = QHBoxLayout(self)
        self.top_layout.addLayout(self.slice_view_layout)
        self.top_layout.addWidget(self.volume_view)

        # connect signals/slots
        self.slice_view.slice_changed[int].connect(self.sliceChanged)
        self.slice_view_slider.valueChanged[int].connect(self.slice_view.setSlice)
        self.button_set_left.clicked.connect(self.setLeftVolume)
        self.button_set_right.clicked.connect(self.setRightVolume)

        # initialize VTK
        self.slice_view.Initialize()
        self.slice_view.Start()
        self.volume_view.Initialize()
        self.volume_view.Start()

    
    def sliceChanged(self, slice_nr):
        self.slice_view_slider.setSliderPosition(slice_nr)


    def setLeftVolume(self):
        print("Setting left volume...")
    
    
    def setRightVolume(self):
        print("Setting right volume...")


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
    

    def loadPatient(self, patient_dict):
        volume_file = patient_dict['volume_raw']
        print("laoding", volume_file)
        if volume_file:
            self.slice_view.loadNrrd(volume_file, flip_x_y=True)
            self.volume_view.loadNrrd(volume_file)
            self.slice_view_slider.setRange(
                self.slice_view.min_slice,
                self.slice_view.max_slice
            )
            self.slice_view_slider.setSliderPosition(self.slice_view.slice)

        else:
            self.slice_view.reset()
            self.volume_view.reset()


    def close(self):
        self.slice_view.Finalize()