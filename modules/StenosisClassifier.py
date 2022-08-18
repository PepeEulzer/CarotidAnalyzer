import os
from re import M
from statistics import mean

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QTabWidget
from PyQt5.QtGui import QColor, QPainterPath
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QLineF
import pyqtgraph as pg

from defaults import *

# Override pyqtgraph defaults
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOption('antialias', True)

# quantitative colors
LINE_COLORS = [(27,158,119),
               (217,95,2),
               (117,112,179),
               (231,41,138),
               (102,166,30)]
LINE_COLORS_QT = [QColor(c[0], c[1], c[2]) for c in LINE_COLORS]
LINE_COLORS_VTK = [(c[0]/255, c[1]/255, c[2]/255) for c in LINE_COLORS]


class LineROI(pg.InfiniteLine):
    """
    Draggable horizontal line that stops at branch boundaries in radius plots.
    """
    def __init__(self, plot_id, x_start, x_end, y_pos=None, angle=90, pen=None, movable=False, y_bounds=None,
                 hoverPen=None, label=None, labelOpts=None, name=None):
        self.plot_id = plot_id
        self.x_start = x_start
        self.x_end = x_end
        super().__init__(y_pos, angle, pen, movable, y_bounds, hoverPen, label, labelOpts, name)


    def boundingRect(self):
        if self._boundingRect is None:
            br = self.viewRect()
            if br is None:
                return QRectF()
            
            # add a 4-pixel radius around the line for mouse interaction.
            px = self.pixelLength(direction=pg.Point(1,0), ortho=True)  # get pixel length orthogonal to the line
            if px is None:
                px = 0
            w = (max(4, self.pen.width()/2, self.hoverPen.width()/2)+1) * px
            br.setBottom(-w)
            br.setTop(w)
            br.setRight(self.x_end)
            br.setLeft(self.x_start)
            
            br = br.normalized()
            self._boundingRect = br
            self._line = QLineF(br.right(), 0.0, br.left(), 0.0)
        return self._boundingRect

    def mouseDragEvent(self, ev):
        if self.movable and ev.button() == Qt.LeftButton:
            if ev.isStart():
                self.moving = True
                self.cursorOffset = self.pos() - self.mapToParent(ev.buttonDownPos())
                self.startPosition = self.pos()
            ev.accept()

            if not self.moving:
                return

            new_pos = self.cursorOffset + self.mapToParent(ev.pos())
            new_pos.setX(self.pos().x())
            self.setPos(new_pos)
            self.sigDragged.emit(self)
            if ev.isFinish():
                self.moving = False
                self.sigPositionChangeFinished.emit(self)


class StenosisClassifierTab(QWidget):
    """
    Tab view of a right OR left side carotid for stenosis classification.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.min_branch_len = 20 # minimal length of a branch in mm
        self.branch_cutoff = 2  # length to be cut from branch ends in mm

        self.centerlines = None    # raw centerlines (vtkPolyData)
        self.c_radii_lists = []    # processed centerline radii
        self.c_pos_lists = []      # processed centerline positions
        self.c_arc_lists = []      # processed centerline arc length (cumulated)
        self.c_parent_indices = [] # index tuples for branch parent / branch point

        # model view
        self.model_view = QVTKRenderWindowInteractor(self)
        self.model_view.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1,1,1)
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(0, 0, -100)
        cam.SetFocalPoint(0, 0, 0)
        cam.SetViewUp(0, -1, 0)
        self.model_view.GetRenderWindow().AddRenderer(self.renderer)

        # graph view
        self.widget_lineplots = pg.GraphicsLayoutWidget()
        self.lineplots = []
        self.scatterplots_down = []
        self.scatterplots_up = []

        # combine all in a layout
        self.top_layout = QHBoxLayout(self)
        self.top_layout.addWidget(self.widget_lineplots)
        self.top_layout.addWidget(self.model_view)

        # lumen vtk pipeline
        self.reader_lumen = vtk.vtkSTLReader()
        self.mapper_lumen = vtk.vtkPolyDataMapper()
        self.mapper_lumen.SetInputConnection(self.reader_lumen.GetOutputPort())
        self.actor_lumen = vtk.vtkActor()
        self.actor_lumen.SetMapper(self.mapper_lumen)
        # self.actor_lumen.GetProperty().SetColor(COLOR_LUMEN)
        self.actor_lumen.GetProperty().SetColor(1,1,1)
        self.actor_lumen.GetProperty().SetOpacity(0.7)
        self.actor_lumen.GetProperty().FrontfaceCullingOn()

        # centerline vtk pipeline
        self.reader_centerline = vtk.vtkXMLPolyDataReader()
        self.centerline_actors = []

        # other vtk props
        self.text_patient = vtk.vtkTextActor()
        self.text_patient.SetInput("No lumen or centerlines file found for this side.")
        self.text_patient.SetDisplayPosition(10, 10)
        self.text_patient.GetTextProperty().SetColor(0, 0, 0)
        self.text_patient.GetTextProperty().SetFontSize(20)
        self.renderer.AddActor(self.text_patient)

        # start interactors
        self.model_view.Initialize()
        self.model_view.Start()


    def showEvent(self, event):
        self.model_view.Enable()
        self.model_view.EnableRenderOn()
        super(StenosisClassifierTab, self).showEvent(event)


    def hideEvent(self, event):
        self.model_view.Disable()
        self.model_view.EnableRenderOff()
        super(StenosisClassifierTab, self).hideEvent(event)
    

    def loadModels(self, lumen_file, centerline_file):
        if lumen_file and centerline_file:
            # load lumen
            self.reader_lumen.SetFileName("") # forces a reload
            self.reader_lumen.SetFileName(lumen_file)
            self.reader_lumen.Update()
            self.renderer.AddActor(self.actor_lumen)
            self.text_patient.SetInput(os.path.basename(lumen_file)[:-4])

            # load centerline
            self.reader_centerline.SetFileName("")
            self.reader_centerline.SetFileName(centerline_file)
            self.reader_centerline.Update()
            self.centerlines = self.reader_centerline.GetOutput()
            self.__preprocessCenterlines()
            self.plot_radii()
            self.draw_active_centerlines()

        else:
            # clear all
            self.renderer.RemoveActor(self.actor_lumen)
            for actor in self.centerline_actors:
                self.renderer.RemoveActor(actor)
            self.centerline_actors = []
            self.centerlines = None
            self.text_patient.SetInput("No lumen or centerlines file found for this side.")

        # reset scene and render
        self.renderer.ResetCamera()
        self.model_view.GetRenderWindow().Render()

    def __preprocessCenterlines(self):
        # lists for each line in centerlines
        # lines are ordered source->outlet
        self.c_pos_lists = []      # 3xn numpy arrays with point positions
        self.c_arc_lists = []      # 1xn numpy arrays with arc length along centerline (accumulated)
        self.c_radii_lists = []    # 1xn numpy arrays with maximal inscribed sphere radius
        self.c_parent_indices = [] # tuple per list: (parent idx, branch point idx)

        # iterate all (global) lines
        # each line is a vtkIdList containing point ids in the right order
        l = self.centerlines.GetLines()        
        l.InitTraversal()
        for i in range(l.GetNumberOfCells()):
            pointIds = vtk.vtkIdList()
            l.GetNextCell(pointIds)

            # retrieve position data
            points = vtk.vtkPoints()
            self.centerlines.GetPoints().GetPoints(pointIds, points)
            p = vtk_to_numpy(points.GetData())

            # calculate arc len
            arc = p - np.roll(p, 1, axis=0)
            arc = np.sqrt((arc*arc).sum(axis=1))
            arc[0] = 0
            arc = np.cumsum(arc)

            # retrieve radius data
            radii_flat = vtk_to_numpy(self.centerlines.GetPointData().GetArray('MaximumInscribedSphereRadius'))
            r = np.zeros(pointIds.GetNumberOfIds())
            for j in range(pointIds.GetNumberOfIds()):
                r[j] = radii_flat[pointIds.GetId(j)]

            # add to centerlines
            self.c_pos_lists.append(p)
            self.c_arc_lists.append(arc)
            self.c_radii_lists.append(r)
            self.c_parent_indices.append((i,0)) # points to own origin

        # cleanup branch overlaps
        # (otherwise each line starts at the inlet)
        for i in range(0, len(self.c_pos_lists)):
            for j in range(i+1, len(self.c_pos_lists)):
                len0 = self.c_pos_lists[i].shape[0]
                len1 = self.c_pos_lists[j].shape[0]
                if len0 < len1:
                    overlap_mask = np.not_equal(self.c_pos_lists[i], self.c_pos_lists[j][:len0])
                else:
                    overlap_mask = np.not_equal(self.c_pos_lists[i][:len1], self.c_pos_lists[j])
                overlap_mask = np.all(overlap_mask, axis=1) # AND over tuples
                split_index = np.searchsorted(overlap_mask, True) # first position where lines diverge

                if split_index <= 0:
                    continue # no new parent was found

                # save parent and position
                self.c_parent_indices[j] = (i,split_index)
                
                # clip line to remove overlaps
                self.c_pos_lists[j] = self.c_pos_lists[j][split_index:]
                self.c_arc_lists[j] = self.c_arc_lists[j][split_index:]
                self.c_radii_lists[j] = self.c_radii_lists[j][split_index:]

        # remove branches below the minimum length
        for i in range(len(self.c_arc_lists)-1, -1, -1):
            if self.c_arc_lists[i][-1] - self.c_arc_lists[i][0] < self.min_branch_len:
                del self.c_pos_lists[i]
                del self.c_arc_lists[i]
                del self.c_radii_lists[i]
                del self.c_parent_indices[i]

        # clip branch ends
        for i in range(len(self.c_arc_lists)):
            start = self.c_arc_lists[i][0] + self.branch_cutoff
            end = self.c_arc_lists[i][-1] - self.branch_cutoff
            clip_ids = np.searchsorted(self.c_arc_lists[i], [start, end])
            self.c_pos_lists[i] = self.c_pos_lists[i][clip_ids[0]:clip_ids[1]]
            self.c_arc_lists[i] = self.c_arc_lists[i][clip_ids[0]:clip_ids[1]]
            self.c_radii_lists[i] = self.c_radii_lists[i][clip_ids[0]:clip_ids[1]]

    
    def plot_radii(self):
        self.widget_lineplots.clear()
        self.lineplots = []
        self.scatterplots_down = []
        self.scatterplots_up = []
        
        for i in range(len(self.c_radii_lists)):
            lineplot = self.widget_lineplots.addPlot()
            lineplot.setLabel('left', "Minimal Radius (mm)")
            lineplot.showGrid(x=False, y=True, alpha=0.2)
            self.lineplots.append(lineplot)

            s = lineplot.plot([], [], pen=None, symbolBrush=(255, 0, 0))
            self.scatterplots_down.append(s)
            s = lineplot.plot([], [], pen=None, symbolBrush=(0, 255, 0))
            self.scatterplots_up.append(s)

            # draw radius lineplot
            lineplot.plot(x=self.c_arc_lists[i], y=self.c_radii_lists[i], pen=LINE_COLORS_QT[i])

            # mark origin of subbranches
            subbranch_ids = sorted([y for x,y in self.c_parent_indices if x==i and y!=0])
            for id in subbranch_ids:
                p = self.c_arc_lists[i][id]
                lineplot.addItem(pg.InfiniteLine(pos=p, angle=90, pen=LINE_COLORS_QT[i+1]))

            # draw horizontal sliders
            subbranch_ids.insert(0, 0) # line start id
            subbranch_ids.append(len(self.c_arc_lists[i])-1) # line end id
            for j in range(len(subbranch_ids)-1):
                id0 = subbranch_ids[j]
                id1 = subbranch_ids[j+1]
                x_min = self.c_arc_lists[i][id0]
                x_max = self.c_arc_lists[i][id1]
                mean_y = np.mean(self.c_radii_lists[i][id0:id1])
                min_y = np.min(self.c_radii_lists[i][id0:id1])
                min_y -= 0.01 * (mean_y-min_y) # small offset below lowest value
                selection_line = LineROI(i, x_min, x_max, y_pos=min_y, y_bounds=[min_y, mean_y], angle=0, movable=True)
                lineplot.addItem(selection_line)
                selection_line.sigPositionChanged[object].connect(self.lineROIposChanged)

            # next graph
            self.widget_lineplots.nextRow()

        # link axes
        for i in range(1, len(self.lineplots)):
            self.lineplots[i].setXLink(self.lineplots[0])
            #self.lineplots[i].setYLink(self.lineplots[0])

        # set label
        self.lineplots[-1].setLabel('bottom', "Branch Length (mm)")

    
    def draw_active_centerlines(self):
        # clear any existing centerline actors
        for actor in self.centerline_actors:
            self.renderer.RemoveActor(actor)
        self.centerline_actors = []
        
        for i in range(len(self.c_pos_lists)):
            points = vtk.vtkPoints()
            for p in self.c_pos_lists[i].tolist():
                points.InsertNextPoint(p)
            
            lines = vtk.vtkCellArray()
            for j in range(points.GetNumberOfPoints()-1):
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0,j)
                line.GetPointIds().SetId(1, j+1)
                lines.InsertNextCell(line)

            polyData = vtk.vtkPolyData()
            polyData.SetPoints(points)
            polyData.SetLines(lines)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polyData)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(LINE_COLORS_VTK[i])
            actor.GetProperty().SetLineWidth(3)
            actor.GetProperty().RenderLinesAsTubesOn()

            self.renderer.AddActor(actor)
            self.centerline_actors.append(actor)


    def lineROIposChanged(self, lineROI):
        arc_lengths = self.c_arc_lists[lineROI.plot_id]
        r_thresh = lineROI.getYPos()

        start_index = np.searchsorted(arc_lengths, lineROI.x_start)
        end_index = np.searchsorted(arc_lengths, lineROI.x_end) + 1
        radii = self.c_radii_lists[lineROI.plot_id][start_index:end_index]
        radii_ranges = np.where(radii < r_thresh, 0, 1)
        radii_ranges[-1] = 1 # closes open ends
        radii_ranges = radii_ranges - np.roll(radii_ranges, 1)
        indices_up = np.where(radii_ranges == 1)[0] + start_index
        indices_down = np.where(radii_ranges == -1)[0] + start_index

        # update debugging plots
        down_plot = self.scatterplots_down[lineROI.plot_id]
        down_plot.setData(arc_lengths[indices_down], np.full(indices_down.size, r_thresh))
        up_plot = self.scatterplots_up[lineROI.plot_id]
        up_plot.setData(arc_lengths[indices_up], np.full(indices_up.size, r_thresh))


    def close(self):
        self.model_view.Finalize()



class StenosisClassifier(QTabWidget):
    """
    Visualization module for analyzing stenoses in vessel trees.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.patient_dict = None

        self.classifier_module_left = StenosisClassifierTab()
        self.classfifier_module_right = StenosisClassifierTab()

        self.addTab(self.classfifier_module_right, "Right")
        self.addTab(self.classifier_module_left, "Left")


    def loadPatient(self, patient_dict):
        self.patient_dict = patient_dict
        self.classfifier_module_right.loadModels(
            patient_dict['lumen_model_right'], patient_dict['centerlines_right'])
        self.classifier_module_left.loadModels(
            patient_dict['lumen_model_left'], patient_dict['centerlines_left'])


    def close(self):
        self.classfifier_module_right.close()
        self.classifier_module_left.close()