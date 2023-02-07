# Carotid Analyzer

A full pipeline for cropping, model extraction, centerline computation, and interactive visualization of carotid artery geometries.

## Files

- `modules` All module widgets and associated classes are contained here.
  - `CenterlineModule.py` Module for generating centerlines.
  - `CropModule.py` Module for cropping CTA volumes.
  - `Interactors.py` Image and 3D interactors shared across modules.
  - `Predictor.py` CNN for plaque/lumen label prediction.
  - `SegmentationModule.py` Module for segmenting cropped images.
  - `StenosisClassifier.py` Module for interactive stenosis classification.
- `scripts` Additional scripts for testing purposes, *not* referenced in the application.
- `ui` UI and resource source files for Qt Designer, *not* referenced in the application.
  - `resources` Contains applications icons etc.
  - `mainwindow.ui` Qt Designer UI file.
  - `resources.qrc` Qt Designer resource file.
- `CarotidAnalyzer.py` Main application, run this for execution.
- `defaults.py` Global constants (colors, symbols...)
- `mainwindow_ui.py` Compiled UI file.
- `models.zip` The carotid bifurcation model database. Can be used as a target folder for the application.
- `resources_rc.py` Compiled resource file.
- `seg_model_weights.pth` Trained CNN weights for carotid lumen and plaque labelling.

## Setup

### Dependencies

- Python 3.10
- numpy 1.23
- pyqt 5.15 (GUI)
- pyqtgraph 0.13 (graphs)
- vtk 9.1 (rendering)
- vmtk 1.5 (centerline computation)
- pytorch 1.13 with monai and scikit-image (segmentation prediction)
- pydicom 2.3 with gdcm (for compressed DICOM I/O)
- pynrrd 0.4 (vtk can only read not write nrrd files)

### Setup with Anaconda

With a CUDA-enabled GPU:

```bash
conda create -n CarotidAnalyzer pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

> **or** install Pytorch for CPU only:
>
> ```bash
> conda create -n CarotidAnalyzer pytorch torchvision torchaudio cpuonly -c pytorch
> ```

Then, install the other packages:

```bash
conda activate CarotidAnalyzer
conda install -c conda-forge monai pyqt pyqtgraph vtk vmtk pydicom pynrrd
conda install scikit-image
pip install python-gdcm
```

The main application can now be run and modified.

## First Steps

1. Use `File -> Set Working Directory` to chose the folder for the patient database. Each case receives a named folder. Module input/output is saved using industry standard formats, so they can be easily externally accessed. For example, the segmentation files can be opened and edited with 3D Slicer.
2. Any existing cases will be shown in the data inspector module. Double-click a case to load it or choose `Load Selected Patient`.
3. To import new cases, use `File -> Load New DICOM` to create a new case subfolder and import a DICOM series (should be an axially resolved head/neck CTA). Choose the folder containing the series. Compressed DICOM files are handled by pydicom with numpy and GDCM, which enables import of most JPEG compression formats. See [this list](https://pydicom.github.io/pydicom/stable/old/image_data_handlers.html#guide-compressed) for a complete overview of supported formats.
4. The pipeline can now be used on the new data. The application will ask if the full volume should be saved or only temporalily loaded. Saving full volumes may take 100-200 MB of disk space. If you do not intend to change the crop region later, saving can be omitted.

## Implementing Extensions

Extension modules that are a subclass of [QWidget](https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QWidget.html) can be integrated directly, analogous to the existing modules.

## GUI Dev with Qt Designer

The `ui` folder is not visible to the application! Do not reference any resources within it directly. The UI and resources are compiled to python files which can be directly imported.

For GUI development with the Qt Designer also install the pyqt5-tools

```bash
conda activate CarotidAnalyzer
pip install pyqt5-tools
```

Run the Qt Designer with the CarotidAnalyzer environment active

```bash
designer
```

Modify UI `.ui` and Qt resource `.qrc` files. On Windows, compile them with

```bash
cd C:\Users\<username>\.conda\envs\CarotidAnalyzer\Scripts
pyuic5.exe <your path>\carotidanalyzer\ui\mainwindow.ui -o <your path>\carotidanalyzer\mainwindow_ui.py
pyrcc5.exe <your path>\carotidanalyzer\ui\resources.qrc -o <your path>\carotidanalyzer\resources_rc.py
```

On other platforms, just use the `pyuic5`/`pyrcc5` commands (no .exe) with the CarotidAnalyzer environment active.