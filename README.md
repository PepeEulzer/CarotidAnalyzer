# Carotid Analyzer

A full pipeline for cropping, segmentation, centerline computation, and interactive visualization of carotid artery geometries.

### Setup

Main dependencies
- Python 3.6 (requirement for vmtk)
- numpy 1.12
- pyqt 5.9
- itk 4.13
- vtk 8.1
- vmtk 1.4

Install using Anaconda:

```
conda create -n CarotidAnalyzer -c vmtk python=3.6 pyqt numpy itk vtk vmtk
conda activate CarotidAnalyzer
conda install -c conda-forge pynrrd pyqtgraph
```

The main application can now be used and edited. Extension modules can also be developed and integrated. The following sections are only relevant if the core GUI (not the GUI of extension modules) needs to be changed or the segmentation CNN needs to be run.



#### GUI Dev with Qt Designer

The `ui` folder is not visible to the application! Do not reference any resources within it directly. The UI and resources are compiled to python files which can be directly imported. 

For GUI development with the Qt Designer also install the pyqt5-tools

```
conda activate CarotidAnalyzer
conda install pip
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



#### Install CNN inference dependencies

For creating segmentation mask predictions with the CNN install the following environment. Use a separate environment, as this requires Python >= 3.7 and thus does not work with the vmtk.

Pytorch CUDA (with CUDA-enabled NVIDIA card):

```
conda create -n CarotidInference -c pytorch pytorch torchvision torchaudio cudatoolkit=11.3
```

> **Or** Pytorch CPU:
>
>```
>conda install pytorch torchvision torchaudio cpuonly -c pytorch
>```

Then, install Monai, skimage, pynrrd:

```
conda activate CarotidInference
pip install monai
conda install -c conda-forge skimage pynrrd 
```

This allows using the *Inference.py* script, which updates the segmentation predictions in the database. It does not modify the segmentations visible to the user and is only used if a new prediction is requested (overwrites any current segmentation).