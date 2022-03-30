import vtk
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
        self.image = None
        self.crop_image_left = None
        self.crop_image_right = None
        
        self.box_left_source = vtk.vtkCubeSource()
        self.box_left_mapper = vtk.vtkPolyDataMapper()
        self.box_left_mapper.SetInputConnection(self.box_left_source.GetOutputPort())
        self.box_left_actor = vtk.vtkActor()
        self.box_left_actor.SetMapper(self.box_left_mapper)

        self.box_right_source = vtk.vtkCubeSource()
        self.box_right_mapper = vtk.vtkPolyDataMapper()
        self.box_right_mapper.SetInputConnection(self.box_right_source.GetOutputPort())
        self.box_right_actor = vtk.vtkActor()
        self.box_right_actor.SetMapper(self.box_right_mapper)
        
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
        if not volume_file:
            self.image = None
            self.slice_view.reset()
            self.volume_view.reset()
            return

        # load raw volume
        reader = vmtkscripts.vmtkImageReader()
        reader.InputFileName = volume_file
        reader.Execute()
        self.image = reader.Image
        self.volume_view.setImage(self.image)
        self.slice_view.setImage(self.image)
        self.slice_view_slider.setRange(
            self.slice_view.min_slice,
            self.slice_view.max_slice
        )
        self.slice_view_slider.setSliderPosition(self.slice_view.slice)

        # load left volume
        left_volume_file = patient_dict['volume_left']
        if left_volume_file:
            reader.InputFileName = left_volume_file
            reader.Execute()
            img = reader.Image
            o_x, o_y, o_z = img.GetOrigin()
            s_x, s_y, s_z = img.GetSpacing()
            x, y, z = img.GetDimensions()
            self.box_left_source.SetBounds(
                        -o_x - s_x * x, -o_x,
                        -o_y - s_y * y, -o_y,  
                         o_z,  o_z + s_z * z
                         )
            self.volume_view.renderer.AddActor(self.box_left_actor)
            self.volume_view.renderer.ResetCamera()
        else:
            self.volume_view.renderer.RemoveActor(self.box_left_actor)

        # load right volume
        # right_volume_file = patient_dict['volume_right']
        # if right_volume_file:
        #     reader.InputFileName = right_volume_file
        #     reader.Execute()
        #     img = reader.Image
        #     self.box_right_source.SetBounds(img.GetExtent())
        #     self.volume_view.renderer.AddActor(self.box_right_actor)
        # else:
        #     self.volume_view.renderer.RemoveActor(self.box_right_actor)




    def close(self):
        self.slice_view.Finalize()
        self.volume_view.Finalize()