# Carotid Analyzer



### Setup

Using anaconda

```
conda create -n CarotidAnalyzer -c conda-forge pyqtgraph=0.12 pyqt
conda activate CarotidAnalyzer
conda install -c vmtk vtk itk vmtk
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
pyuic5.exe C:\Git\test\ui\mainwindow.ui -o C:\Git\test\mainwindow_ui.py
pyrcc5.exe C:\Git\test\ui\resources.qrc -o C:\Git\test\resources_rc.py
```

Adapt to your Git path. On other platforms, just use the `pyuic5`/`pyrcc5` commands (no .exe) with the CarotidAnalyzer environment active.

