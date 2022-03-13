from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vmtk import vmtkscripts

class VTKImageSliceInteractor(QVTKRenderWindowInteractor):
    """
    Displays an image view of a volume slice.
    Interactions: Pan, zoom, scroll.
    """
    
    
    # TODO use a vtkImageViewer2 instead?
    # https://kitware.github.io/vtk-examples/site/Cxx/IO/ReadDICOMSeries/


    def __init__(self, parent=None):
        super().__init__(parent)
        self.SetInteractorStyle(vtk.vtkInteractorStyleImage())
        self.image = vtk.vtkImageData()
        
        # lut for normal CTA display (gray map)
        self.CTA_lut = vtk.vtkLookupTable()
        self.CTA_lut.SetNumberOfTableValues(255)
        self.CTA_lut.SetTableRange(-800, 1200)
        self.CTA_lut.SetSaturationRange(0, 0)
        self.CTA_lut.SetHueRange(0, 0)
        self.CTA_lut.SetValueRange(0.0, 1.0)
        self.CTA_lut.Build()

        # build image mapper, actor pipeline
        self.image_mapper = vtk.vtkDataSetMapper()
        self.image_mapper.SetInputData(self.image)
        self.image_mapper.SetUseLookupTableScalarRange(1)
        self.image_mapper.SetLookupTable(self.CTA_lut)

        self.image_actor = vtk.vtkActor()
        self.image_actor.SetMapper(self.image_mapper)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0,0,0)
        self.renderer.AddActor(self.image_actor)
        #self.rendererR.GetActiveCamera().SetDistance(0.04)
        self.GetRenderWindow().AddRenderer(self.renderer)


    def loadNrrd(self, path):
        reader = vmtkscripts.vmtkImageReader()
        reader.InputFileName = path
        reader.Execute()
        self.image = reader.Image
        self.image_mapper.SetInputData(self.image)
        self.renderer.ResetCamera()
        self.GetRenderWindow().Render()



class CropModule(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.slice_view = VTKImageSliceInteractor(self)


        self.slice_view_layout = QVBoxLayout(self)
        self.slice_view_layout.setContentsMargins(0,0,0,0)
        self.slice_view_layout.addWidget(self.slice_view)

        self.slice_view.Initialize()
        self.slice_view.Start()


    def showEvent(self, event):
        print("show event on crop module")
        self.slice_view.Enable()
        self.slice_view.EnableRenderOn()
        super(CropModule, self).showEvent(event)


    def hideEvent(self, event):
        print("hide event on crop module")
        self.slice_view.Disable()
        self.slice_view.EnableRenderOff()
        super(CropModule, self).hideEvent(event)
    

    def load_patient(self, patient_dict):
        print("Crop module loading " + patient_dict['patient_ID'])
        self.slice_view.loadNrrd(patient_dict['volume_left'])