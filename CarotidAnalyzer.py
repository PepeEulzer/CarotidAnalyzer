import sys

from PyQt5.QtWidgets import (
    QApplication, QDialog, QMainWindow, QMessageBox
)

from mainwindow_ui import Ui_MainWindow

class CarotidAnalyzer(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        
        # connect signals to slots
        self.action_quit.triggered.connect(self.close)
        self.action_data_inspector.triggered[bool].connect(self.view_data_inspector)

    def view_data_inspector(self, on:bool):
        if on:
            self.dock_data_inspector.show()
        else:
            self.dock_data_inspector.close()

        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setOrganizationName("VisGroup Uni Jena")
    app.setOrganizationDomain("vis.uni-jena.de")
    app.setApplicationName("CarotidAnalyzer")
    app.setStyle("Fusion")
    win = CarotidAnalyzer()
    win.show()
    sys.exit(app.exec())