import numpy as np
import vtk
from PyQt5.QtCore import pyqtSignal
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vmtk import vmtkscripts

from defaults import *

class ImageSliceInteractor(QVTKRenderWindowInteractor):
    """
    Displays an image view of a volume slice in z-direction.
    Interactions: Pan, zoom, scroll slices.
    """
    slice_changed = pyqtSignal(int)
    def __init__(self, parent=None):
        super().__init__(parent)
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
        self.GetRenderWindow().AddRenderer(self.renderer)

        # set the amera to the "backside" (view in positive z)
        # for correct patient orientation
        cam = self.renderer.GetActiveCamera()
        cam.ParallelProjectionOn()
        proj_dir = np.array(cam.GetDirectionOfProjection())
        proj_dist = np.array(cam.GetDistance())
        pos = np.array(cam.GetPosition())
        pos = pos + 2 * proj_dist * proj_dir
        cam.SetPosition(pos)


    def setSlice(self, slice_nr):
        self.slice = slice_nr
        self.image_mapper.SetSliceNumber(slice_nr)
        self.GetRenderWindow().Render()
        self.slice_changed.emit(self.slice)


    def mouseWheelForward(self, obj, event):
        if self.slice < self.max_slice:
            self.slice += 1
            self.image_mapper.SetSliceNumber(self.slice)
            self.GetRenderWindow().Render()
            self.slice_changed.emit(self.slice)


    def mouseWheelBackward(self, obj, event):
        if self.slice > self.min_slice:
            self.slice -= 1
            self.image_mapper.SetSliceNumber(self.slice)
            self.GetRenderWindow().Render()
            self.slice_changed.emit(self.slice)


    def loadNrrd(self, path, flip_x_y=False):
        reader = vmtkscripts.vmtkImageReader()
        reader.InputFileName = path
        reader.Execute()
        self.image = reader.Image
        if flip_x_y:
            sx, sy, sz = self.image.GetSpacing()
            ox, oy, oz = self.image.GetOrigin()
            self.image.SetSpacing(-sx, -sy, sz)
            self.image.SetOrigin(-ox, -oy, oz)
        self.image_mapper.SetInputData(self.image)
        self.min_slice = self.image_mapper.GetSliceNumberMinValue()
        self.max_slice = self.image_mapper.GetSliceNumberMaxValue()
        self.slice = self.min_slice

        # re-focus the camera
        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().SetClippingRange(10, 1000)
        self.GetRenderWindow().Render()

    
    def reset(self):
        self.image = vtk.vtkImageData()
        self.image_mapper.SetInputData(self.image)
        self.min_slice = 0
        self.max_slice = 0
        self.slice = 0
        self.GetRenderWindow().Render()



class IsosurfaceInteractor(QVTKRenderWindowInteractor):
    """
    Displays a 3D view of an isosurface reconstructed from a segmentation.
    Interactions: Rotate, Zoom, Translate.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.label_map = vtk.vtkImageData()

        # build isosurface, mapper, actor pipeline
        self.padding = vtk.vtkImageConstantPad()
        self.padding.SetConstant(0)
        self.padding.SetInputData(self.label_map)

        self.marching_lumen = vtk.vtkDiscreteMarchingCubes()
        self.marching_lumen.SetInputConnection(self.padding.GetOutputPort())
        self.marching_lumen.GenerateValues(1, 2, 2)
        self.smoother_lumen = vtk.vtkWindowedSincPolyDataFilter()
        self.smoother_lumen.SetInputConnection(self.marching_lumen.GetOutputPort())
        self.smoother_lumen.SetNumberOfIterations(20)
        self.smoother_lumen.SetPassBand(0.005)
        self.mapper_lumen = vtk.vtkPolyDataMapper()
        self.mapper_lumen.SetInputConnection(self.smoother_lumen.GetOutputPort())
        self.mapper_lumen.ScalarVisibilityOff()
        self.actor_lumen = vtk.vtkActor()
        self.actor_lumen.GetProperty().SetColor(COLOR_LUMEN)
        self.actor_lumen.SetMapper(self.mapper_lumen)

        self.marching_plaque = vtk.vtkDiscreteMarchingCubes()
        self.marching_plaque.SetInputConnection(self.padding.GetOutputPort())
        self.marching_plaque.GenerateValues(1, 1, 1)
        self.smoother_plaque = vtk.vtkWindowedSincPolyDataFilter()
        self.smoother_plaque.SetInputConnection(self.marching_plaque.GetOutputPort())
        self.smoother_plaque.SetNumberOfIterations(20)
        self.smoother_plaque.SetPassBand(0.005)
        self.mapper_plaque = vtk.vtkPolyDataMapper()
        self.mapper_plaque.SetInputConnection(self.smoother_plaque.GetOutputPort())
        self.mapper_plaque.ScalarVisibilityOff()
        self.actor_plaque = vtk.vtkActor()
        self.actor_plaque.GetProperty().SetColor(COLOR_PLAQUE)
        self.actor_plaque.SetMapper(self.mapper_plaque)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1,1,1)
        self.GetRenderWindow().AddRenderer(self.renderer)


    def loadNrrd(self, path):
        # read the file
        reader = vmtkscripts.vmtkImageReader()
        reader.InputFileName = path
        reader.Execute()
        self.label_map = reader.Image
        
        # convert to check if plaque actor is necessary
        img_to_numpy = vmtkscripts.vmtkImageToNumpy()
        img_to_numpy.Image = self.label_map
        img_to_numpy.Execute()
        
        # add padding
        extent = np.array(self.label_map.GetExtent())
        extent += np.array([-1, 1, -1, 1, -1, 1])
        self.padding.SetInputData(self.label_map)
        self.padding.SetOutputWholeExtent(extent)
        
        # update the scene (pipeline triggers automatically)
        self.renderer.AddActor(self.actor_lumen)
        if 1.0 in img_to_numpy.ArrayDict['PointData']['ImageScalars']:
            self.renderer.AddActor(self.actor_plaque)
        else:
            self.renderer.RemoveActor(self.actor_plaque)
        self.renderer.ResetCamera()
        self.GetRenderWindow().Render()


    def reset(self):
        self.label_map = vtk.vtkImageData()
        self.renderer.RemoveActor(self.actor_lumen)
        self.renderer.RemoveActor(self.actor_plaque)
        self.GetRenderWindow().Render()



class VolumeRenderingInteractor(QVTKRenderWindowInteractor):
    """
    Displays a 3D view of an CTA volume rendering.
    Interactions: Rotate, Zoom, Translate.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.image = vtk.vtkImageData()

        # ====================================================================
        # Initialization: CTA context visualization
        # ====================================================================
        self.CTA_opacity = 50 # value in %
        self.CTA_threshold = 170 # value in HU [70, 270]

        # CTA mapper
        self.CTA_mapper = vtk.vtkGPUVolumeRayCastMapper()
        self.CTA_mapper.SetBlendModeToComposite()
        self.CTA_mapper.SetLockSampleDistanceToInputSpacing(False)
        self.CTA_mapper.SetAutoAdjustSampleDistances(True)
        self.CTA_mapper.SetUseJittering(True)
                
        # transfer functions
        color_function = vtk.vtkColorTransferFunction()
        color_function.AddRGBPoint(-2048.0, 0.00, 0.00, 0.00, 0.5, 0.0)
        color_function.AddRGBPoint(   25.0, 0.00, 0.00, 0.00, 0.5, 0.0)
        color_function.AddRGBPoint(  120.0, 0.62, 0.00, 0.02, 0.5, 0.0)
        color_function.AddRGBPoint(  160.0, 0.91, 0.45, 0.00, 0.5, 0.0)
        color_function.AddRGBPoint(  190.0, 0.97, 0.81, 0.61, 0.5, 0.0)
        color_function.AddRGBPoint(  240.0, 0.91, 0.91, 1.00, 0.5, 0.0)
        color_function.AddRGBPoint( 3600.0, 1.00, 1.00, 1.00, 0.5, 0.0)
        
        self.opacity_function = vtk.vtkPiecewiseFunction()        
        self.updateOpacityFunction()
       
        self.gradient_function = vtk.vtkPiecewiseFunction()
        self.updateGradientFunction()
        
        volume_prop = vtk.vtkVolumeProperty()
        volume_prop.SetIndependentComponents(True)
        volume_prop.SetColor(color_function)
        volume_prop.SetScalarOpacity(self.opacity_function)
        volume_prop.SetGradientOpacity(self.gradient_function)
        volume_prop.SetInterpolationTypeToLinear()
        volume_prop.ShadeOn()
        volume_prop.SetAmbient(0.2)
        volume_prop.SetDiffuse(0.95)
        volume_prop.SetSpecular(0.1)
        volume_prop.SetSpecularPower(10.0)
        
        self.CTA_volume = vtk.vtkVolume()
        self.CTA_volume.SetProperty(volume_prop)
        self.CTA_volume.SetMapper(self.CTA_mapper)

        # ====================================================================
        # Initialization: Renderer
        # ====================================================================
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1,1,1)
        self.GetRenderWindow().AddRenderer(self.renderer)


    def updateOpacityFunction(self):
        self.opacity_function.Initialize()
        
        # determine control point opacities
        op1 = self.CTA_opacity * 0.01  # [0, 100] -> [0, 1]
        op2 = self.CTA_opacity * 0.006 # [0, 100] -> [0, 0.6]
        
        # threshold should have a value in [70, 270] (HU)
        th0 = self.CTA_threshold - 30
        th1 = self.CTA_threshold
        th2 = self.CTA_threshold + 70
        th3 = self.CTA_threshold + 130
    
        self.opacity_function.AddPoint(-2048.0, 0.0, 0.5, 0.0)
        self.opacity_function.AddPoint(    th0, 0.0, 0.4, 0.0)
        self.opacity_function.AddPoint(    th1, op1, 0.5, 0.1)
        self.opacity_function.AddPoint(    th2, op2, 0.7, 0.1)
        self.opacity_function.AddPoint(    th3, 0.0, 0.5, 0.0)
        self.opacity_function.AddPoint( 3600.0, 0.0, 0.5, 0.0)
        
    
    def updateGradientFunction(self):
        self.gradient_function.Initialize()
        
        op1 = self.CTA_opacity * 0.01 - 0.5    # [0, 100] -> [-0.5, 0.5]
        op2 = self.CTA_opacity * 0.0084 - 0.24 # [0, 100] -> [-0.24, 0.6]
        
        # threshold should have a value in [70, 270] (HU)
        th0 = self.CTA_threshold - 210
        th1 = self.CTA_threshold - 130
        th2 = self.CTA_threshold - 90
        th3 = self.CTA_threshold - 20
        th4 = self.CTA_threshold + 10
        th5 = self.CTA_threshold + 130
        
        self.gradient_function.AddPoint(-2048.0, op1, 0.5, 0.0)
        self.gradient_function.AddPoint(    th0, op1, 0.4, 0.0)
        self.gradient_function.AddPoint(    th1, op2, 0.5, 0.1)
        self.gradient_function.AddPoint(    th2, 1.00, 0.7, 0.1)
        self.gradient_function.AddPoint(    th3, 1.00, 0.5, 0.0)
        self.gradient_function.AddPoint(    th4, op2, 0.5, 0.0)
        self.gradient_function.AddPoint(    th5, op1, 0.5, 0.0)
        self.gradient_function.AddPoint( 3600.0, op1, 0.5, 0.0)


    def loadNrrd(self, path):
        reader = vmtkscripts.vmtkImageReader()
        reader.InputFileName = path
        reader.Execute()
        self.image = reader.Image
        self.CTA_mapper.SetInputData(self.image)

        self.renderer.AddVolume(self.CTA_volume)
        self.renderer.ResetCamera()
        self.GetRenderWindow().Render()


    # def setImage(self, image):
    #     self.image = image
    #     self.CTA_mapper.SetInputData(image)
    #     self.renderer.AddVolume(self.CTA_volume)
    #     self.renderer.ResetCamera()
    #     self.GetRenderWindow().Render()

    
    def reset(self):
        self.image = vtk.vtkImageData()
        self.renderer.RemoveVolume(self.CTA_volume)
        self.GetRenderWindow().Render()



