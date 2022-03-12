# Carotid Analyzer



### Setup

Dependencies
- Python 3.6 (requirement for vmtk)
- numpy 1.12
- pyqt 5.9
- itk 4.13
- vtk 8.1
- vmtk 1.4

Install using Anaconda

```
conda create -n CarotidAnalyzer -c vmtk python=3.6 pyqt numpy itk vtk vmtk
```



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

