# Carotid Analyzer

A full pipeline for cropping, segmentation, centerline computation, and interactive visualization of carotid artery geometries.

### Setup

Main dependencies
- Python 3.10
- pytorch 1.13
- numpy 1.23
- pyqt 5.15
- vtk 9.1

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
conda install -c conda-forge monai pyqt vtk pynrrd pyqtgraph
conda install -c anaconda scikit-image
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
