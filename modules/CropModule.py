import vtk
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QPushButton
from vmtk import vmtkscripts

from defaults import COLOR_LEFT, COLOR_LEFT_HEX, COLOR_RIGHT, COLOR_RIGHT_HEX
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
        
        self.slice_view = ImageSliceInteractor(self)
        self.volume_view = VolumeRenderingInteractor(self)
        self.slice_view_slider = QSlider(Qt.Horizontal)
        self.button_set_left = QPushButton("Set Left Volume")
        self.button_set_left.setStyleSheet("background-color:" + COLOR_LEFT_HEX)
        self.button_set_right = QPushButton("Set Right Volume")
        self.button_set_right.setStyleSheet("background-color:" + COLOR_RIGHT_HEX)

        self.cut_left_actor = self.__getCutActor(self.box_left_source.GetOutputPort(), COLOR_LEFT)
        self.cut_right_actor = self.__getCutActor(self.box_right_source.GetOutputPort(), COLOR_RIGHT)

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


    def setLeftVolume(self):
        print("Setting left volume...")
        print("NOT IMPLEMENTED")
    
    
    def setRightVolume(self):
        print("Setting right volume...")
        print("NOT IMPLEMENTED")


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


    def __loadCropVolume(self, filename, box_source, box_actor, cut_actor):
        if filename:
            reader = vmtkscripts.vmtkImageReader()
            reader.InputFileName = filename
            reader.Execute()
            img = reader.Image

            # In the nrrd header, the origin x,y and spacing x,y are mirrored.
            # Might be an artifact of the LPS coordinate system?
            # => Box x,y axes need to be flipped.
            ox, oy, oz = img.GetOrigin()
            sx, sy, sz = img.GetSpacing()
            x, y, z = img.GetDimensions()
            box_source.SetBounds(
                        -ox - sx * x, -ox,
                        -oy - sy * y, -oy,  
                         oz,  oz + sz * z
                         )
            self.volume_view.renderer.AddActor(box_actor)
            self.volume_view.renderer.AddActor(cut_actor)
            self.volume_view.renderer.ResetCamera()
            self.slice_view.renderer.AddActor(cut_actor)
        else:
            self.volume_view.renderer.RemoveActor(box_actor)
            self.volume_view.renderer.RemoveActor(cut_actor)
            self.slice_view.renderer.RemoveActor(cut_actor)

    
    def resetViews(self):
        self.volume_view.renderer.RemoveActor(self.box_left_actor)
        self.volume_view.renderer.RemoveActor(self.cut_left_actor)
        self.slice_view.renderer.RemoveActor(self.cut_left_actor)
        self.volume_view.renderer.RemoveActor(self.box_right_actor)
        self.volume_view.renderer.RemoveActor(self.cut_right_actor)
        self.slice_view.renderer.RemoveActor(self.cut_right_actor)
        self.slice_view.reset()
        self.volume_view.reset()

    def loadPatient(self, patient_dict):
        volume_file = patient_dict['volume_raw']
        if not volume_file:
            self.image = None
            self.resetViews()
            return

        # load raw volume
        reader = vmtkscripts.vmtkImageReader()
        reader.InputFileName = volume_file
        reader.Execute()
        self.image = reader.Image

        # In the nrrd header, the origin x,y are mirrored.
        # => Flip x,y origin positions.
        ox, oy, oz = self.image.GetOrigin()
        self.image.SetOrigin(-ox, -oy, oz)

        # set the volume image in both views
        self.volume_view.setImage(self.image)
        self.slice_view.setImage(self.image)
        self.slice_view_slider.setRange(
            self.slice_view.min_slice,
            self.slice_view.max_slice
        )
        self.slice_view_slider.setSliderPosition(self.slice_view.slice)

        # load crop volume boxes if they exist
        left_volume_file  = patient_dict['volume_left']
        right_volume_file = patient_dict['volume_right']
        self.__loadCropVolume(left_volume_file, self.box_left_source, self.box_left_actor, self.cut_left_actor)
        self.__loadCropVolume(right_volume_file, self.box_right_source, self.box_right_actor, self.cut_right_actor)


    def close(self):
        self.slice_view.Finalize()
        self.volume_view.Finalize()