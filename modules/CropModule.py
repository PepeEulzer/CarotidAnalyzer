from PyQt5.QtWidgets import QWidget

class CropModule(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
    
    def load_patient(self, patient_dict):
        print("Crop module loading " + patient_dict['patient_ID'])