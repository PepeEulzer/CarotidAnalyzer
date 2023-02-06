# Carotid Analyzer

A full pipeline for cropping, model extraction, centerline computation, and interactive visualization of carotid artery geometries.

## Files

```C
carotidanalyzer
├───modules // All module widgets and associated classes are contained here.
│   └───CenterlineModule.py // Module for generating centerlines.
│   └───CropModule.py // Module for cropping CTA volumes.
│   └───Interactors.py // Image and 3D interactors shared across modules.
│   └───Predictor.py // CNN for plaque/lumen label prediction.
│   └───SegmentationModule.py // Module for segmenting cropped images.
│   └───StenosisClassifier.py // Module for interactive stenosis classification.
├───scripts // Additional scripts for testing purposes, NOT referenced in the application.
├───ui // QT Designer UI and resource source files, NOT referenced in the application.
│   └───resources
│   └───mainwindow.ui
│   └───resources.qrc
└───CarotidAnalyzer.py // Main application, run this for execution.
└───defaults.py // Global constants (colors, symbols...)
└───mainwindow_ui.py 
└───models.zip
└───resources_rc.py
└───seg_model_weights.pth
```

## Setup

Main dependencies
- Python 3.10
- numpy 1.23
- pyqt 5.15 (GUI)
- pyqtgraph 0.13 (graphs)
- vtk 9.1 (rendering)
- vmtk 1.5 (centerline computation)
- pytorch 1.13 with monai and scikit-image (segmentation prediction)
- pydicom 2.3 with gdcm (for compressed DICOM I/O)

Install using Anaconda, with CUDA-enabled GPU:
```
conda create -n CarotidAnalyzer pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

> **or** install Pytorch for CPU only
> ```
> conda create -n CarotidAnalyzer pytorch torchvision torchaudio cpuonly -c pytorch
> ```

Then install other packages:
```
conda activate CarotidAnalyzer
conda install -c conda-forge monai pyqt pyqtgraph vtk vmtk pydicom pynrrd
conda install scikit-image
pip install python-gdcm
```

The main application can now be used and edited. Extension modules can also be developed and integrated. The following sections are only relevant if the core GUI (not the GUI of extension modules) needs to be changed.



#### GUI Dev with Qt Designer

The `ui` folder is not visible to the application! Do not reference any resources within it directly. The UI and resources are compiled to python files which can be directly imported. 

For GUI development with the Qt Designer also install the pyqt5-tools

```
conda activate CarotidAnalyzer
pip install pyqt5-tools
```

Run the Qt Designer with the CarotidAnalyzer environment active

```
designer
```

Modify UI `.ui` and Qt resource `.qrc` files. On Windows, compile them with

```
cd C:\Users\<username>\.conda\envs\CarotidAnalyzer\Scripts
pyuic5.exe C:\Git\carotidanalyzer\ui\mainwindow.ui -o C:\Git\carotidanalyzer\mainwindow_ui.py
pyrcc5.exe C:\Git\carotidanalyzer\ui\resources.qrc -o C:\Git\carotidanalyzer\resources_rc.py
```

Adapt to your Git path. On other platforms, just use the `pyuic5`/`pyrcc5` commands (no .exe) with the CarotidAnalyzer environment active.
