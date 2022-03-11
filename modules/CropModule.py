from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vmtk import vmtkscripts

class VTKImageSliceInteractor(QVTKRenderWindowInteractor):
    """
    Displays an image view of a volume slice.
    Interactions: Pan, zoom, scroll.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setInteractorStyle(vtk.vtkInteractorStyleImage())
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
        print(reader.Image)



class CropModule(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.slice_view = VTKImageSliceInteractor(self)


        self.slice_view_layout = QVBoxLayout()
        self.slice_view_layout.addWidget()
    
    def load_patient(self, patient_dict):
        print("Crop module loading " + patient_dict['patient_ID'])
        #self.