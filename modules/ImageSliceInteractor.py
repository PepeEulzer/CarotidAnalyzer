import numpy as np
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vmtk import vmtkscripts

class ImageSliceInteractor(QVTKRenderWindowInteractor):
    """
    Displays an image view of a volume slice in z-direction.
    Interactions: Pan, zoom, scroll slices.
    """
    def __init__(self, slider, parent=None):
        super().__init__(parent)
        self.slider = slider
        self.slider.valueChanged[int].connect(self.slider_moved)
        self.interactor_style = vtk.vtkInteractorStyleImage()
        self.interactor_style.AddObserver("MouseWheelForwardEvent", self.mouseWheelForward)
        self.interactor_style.AddObserver("MouseWheelBackwardEvent", self.mouseWheelBackward)
        self.SetInteractorStyle(self.interactor_style)
        self.slice = 0
        self.min_slice = 0
        self.max_slice = 0
        self.image = vtk.vtkImageData()

        # build image mapper, actor pipeline
        self.image_mapper = vtk.vtkOpenGLImageSliceMapper()
        self.image_mapper.SliceAtFocalPointOff()
        self.image_mapper.SetNumberOfThreads(1)
        self.image_mapper.SetInputData(self.image)

        self.image_actor = vtk.vtkImageActor()
        self.image_actor.SetMapper(self.image_mapper)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0,0,0)
        self.renderer.AddActor(self.image_actor)
        self.renderer.GetActiveCamera().ParallelProjectionOn()
        self.GetRenderWindow().AddRenderer(self.renderer)


    def slider_moved(self, pos):
        self.slice = pos
        self.image_mapper.SetSliceNumber(pos)
        self.GetRenderWindow().Render()


    def mouseWheelForward(self, obj, event):
        if self.slice < self.max_slice:
            self.slice += 1
            self.image_mapper.SetSliceNumber(self.slice)
            self.slider.setSliderPosition(self.slice)
            self.GetRenderWindow().Render()


    def mouseWheelBackward(self, obj, event):
        if self.slice > self.min_slice:
            self.slice -= 1
            self.image_mapper.SetSliceNumber(self.slice)
            self.slider.setSliderPosition(self.slice)
            self.GetRenderWindow().Render()


    def loadNrrd(self, path):
        reader = vmtkscripts.vmtkImageReader()
        reader.InputFileName = path
        reader.Execute()
        self.image = reader.Image
        self.image_mapper.SetInputData(self.image)
        self.min_slice = self.image_mapper.GetSliceNumberMinValue()
        self.max_slice = self.image_mapper.GetSliceNumberMaxValue()
        self.slice = self.min_slice
        self.slider.setRange(self.min_slice, self.max_slice)
        self.slider.setSliderPosition(self.slice)

        # re-focus the camera
        # set it to the "backside" (view in positive z) for correct patient orientation
        # TODO when does this apply?? always?
        self.renderer.ResetCamera()
        cam = self.renderer.GetActiveCamera()
        proj_dir = np.array(cam.GetDirectionOfProjection())
        proj_dist = np.array(cam.GetDistance())
        pos = np.array(cam.GetPosition())
        pos = pos + 2 * proj_dist * proj_dir
        cam.SetPosition(pos)
        cam.SetClippingRange(10, 300)
        self.GetRenderWindow().Render()

    
    def reset(self):
        self.image = vtk.vtkImageData()
        self.image_mapper.SetInputData(self.image)
        self.min_slice = 0
        self.max_slice = 0
        self.slice = 0
        self.slider.setRange(0, 100)
        self.slider.setSliderPosition(0)
        self.GetRenderWindow().Render()