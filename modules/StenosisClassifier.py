import os
import json

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import QWidget, QShortcut, QHBoxLayout, QTabWidget, QGraphicsPathItem
from PyQt5.QtGui import QColor, QPainterPath, QKeySequence
from PyQt5.QtCore import Qt, QRectF, pyqtSignal
import pyqtgraph as pg

from defaults import *

# Override pyqtgraph defaults
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOption('antialias', True)

# quantitative colors
STENOSIS_COLORS = [(27,158,119),
                   (217,95,2),
                   (117,112,179),
                   (231,41,138),
                   (102,166,30),
                   (230,171,2),
                   (166,118,29)]
STENOSIS_COLORS_QT = [QColor(c[0], c[1], c[2]) for c in STENOSIS_COLORS]
STENOSIS_COLORS_QT_H = [QColor(c[0]+30, c[1]+30, c[2]+30) for c in STENOSIS_COLORS]
STENOSIS_COLORS_VTK = [(c[0]/255, c[1]/255, c[2]/255) for c in STENOSIS_COLORS]
STENOSIS_COLORS_VTK_H = [((c[0]+30)/255, (c[1]+30)/255, (c[2]+30)/255) for c in STENOSIS_COLORS]

# default pens/brushes
LINEPLOT_DEFAULT_PEN = pg.mkPen({'color':(0,0,0), 'width':2})
LINEPLOT_WIDE_PEN = pg.mkPen({'color':(0,0,0), 'width':3})
TEXT_BG_BRUSH = pg.mkBrush(255, 255, 240)
Q_COLOR_BLACK = pg.mkColor(0, 0, 0)


class LineROI(pg.InfiniteLine):
    """
    Draggable horizontal line that stops at branch boundaries in diameter plots.
    """
    def __init__(self, plot_id, x_start, x_end, y_pos=None, angle=90, pen=None, movable=False, y_bounds=None,
                 hoverPen=None, label=None, labelOpts=None, span=(0, 1), markers=None, name=None):
        self.plot_id = plot_id
        self.x_start = x_start
        self.x_end = x_end
        super().__init__(y_pos, angle, pen, movable, y_bounds, hoverPen, label, labelOpts, span, markers, name)


    def _computeBoundingRect(self):
        vr = self.viewRect()  # bounds of containing ViewBox mapped to local coords.
        if vr is None:
            return QRectF()
        
        px = self.pixelLength(direction=pg.Point(1,0), ortho=True)  # get pixel length orthogonal to the line
        if px is None:
            px = 0
        pw = max(self.pen.width() / 2, self.hoverPen.width() / 2)
        w = (self._maxMarkerSize + pw + 1) * px
        br = QRectF(vr)
        br.setBottom(-w)
        br.setTop(w)

        length = br.width()
        left = br.left() + length
        right = br.left() + length
        br.setRight(self.x_end)
        br.setLeft(self.x_start)
        br = br.normalized()
        
        vs = self.getViewBox().size()
        
        if self._bounds != br or self._lastViewSize != vs:
            self._bounds = br
            self._lastViewSize = vs
            self.prepareGeometryChange()
        
        self._endPoints = (self.x_start, self.x_end)
        self._lastViewRect = vr
        
        return self._bounds


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



class HoverablePlotItem(pg.PlotItem):
    sigHoverEnter = pyqtSignal(object)
    sigHoverLeave = pyqtSignal(object)
    def __init__(self, plotID, *args):
        super().__init__(*args)
        self.plotID = plotID
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, ev):
        self.sigHoverEnter.emit(self)

    def hoverLeaveEvent(self, ev):
        self.sigHoverLeave.emit(self)



class StenosisAreaItem(QGraphicsPathItem):
    """
    Displays the area of a stenosis in a graph.
    """
    def __init__(self, activate_method, inactivate_method, brush_inactive, brush_active, *args):
        super().__init__(*args)
        self.activate_method = activate_method
        self.inactivate_method = inactivate_method
        self.brush_inactive = brush_inactive
        self.brush_hover = brush_active
        self.setPen(LINEPLOT_DEFAULT_PEN)
        self.setBrush(self.brush_inactive)
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, ev):
        self.setBrush(self.brush_hover)
        self.activate_method()

    def hoverLeaveEvent(self, ev):
        self.setBrush(self.brush_inactive)
        self.inactivate_method()



class StenosisWrapper(object):
    """
    Keeps track of all actors/graph objects needed to display one stenosis.
    """
    def __init__(self, vtk_renderer, lineplot, lineROI,
                 vessel_geometry, pos_array, rad_array, arc_array, start_index, idx1, idx2, colorID):
        
        # reference to holding objects
        self.vtk_renderer = vtk_renderer
        self.lineplot = lineplot
        self.start_index = start_index
        self.colorID = colorID

        # reference to relevant data arrays
        self.pos = pos_array
        self.rad = rad_array
        self.arc = arc_array

        # meta information on stenosis
        self.degree = 0.0
        self.degree_string = "0.0%"
        self.stenosis_arc_len = 0
        self.threshold =lineROI.getYPos()
        self.min_diameter_pos = None
        self.min_diameter_normal = None
        self.ref_diameter_pos = None
        self.ref_diameter_normal = None

        # compute implicit spheres around stenosis region
        clip_function_spheres = vtk.vtkImplicitBoolean()
        clip_function_spheres.SetOperationTypeToUnion()
        max_rad = 2*np.max(self.rad[idx1:idx2])
        for idx in range(idx1, idx2, 10):
            sphere = vtk.vtkSphere()
            sphere.SetCenter(self.pos[idx])
            sphere.SetRadius(max_rad)
            clip_function_spheres.AddFunction(sphere)

        # compute implicit planes to cap stenosis region
        plane1 = vtk.vtkPlane()
        plane1.SetOrigin(self.pos[idx1])
        plane1.SetNormal(np.mean(self.pos[idx1-5:idx1+1] - self.pos[idx1:idx1+6], axis=0))
        plane2 = vtk.vtkPlane()
        plane2.SetOrigin(self.pos[idx2])
        plane2.SetNormal(np.mean(self.pos[idx2:idx2+6] - self.pos[idx2-5:idx2+1], axis=0))

        clip_function = vtk.vtkImplicitBoolean()
        clip_function.SetOperationTypeToIntersection()
        clip_function.AddFunction(clip_function_spheres)
        clip_function.AddFunction(plane1)
        clip_function.AddFunction(plane2)

        self.clipper = vtk.vtkClipPolyData()
        self.clipper.SetInputDataObject(vessel_geometry)
        self.clipper.SetClipFunction(clip_function)
        self.clipper.SetInsideOut(True)
        self.clipper.Update()

        # create 3D stenosis actor
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.clipper.GetOutputPort())
        self.stenosis_actor = vtk.vtkActor()
        self.stenosis_actor.SetMapper(self.mapper)
        self.stenosis_actor.GetProperty().SetColor(STENOSIS_COLORS_VTK[colorID])
        self.vtk_renderer.AddActor(self.stenosis_actor)

        # create 2D stenosis area
        area_x = self.arc[idx1:idx2]
        area_y = 2.0 * self.rad[idx1:idx2]
        path = QPainterPath()
        x1 = self.__getLineIntersection(self.arc[idx1-1], self.arc[idx1], 2.0 * self.rad[idx1-1], 2.0 * self.rad[idx1], self.threshold)
        path.moveTo(x1, self.threshold)
        for i in range(area_x.shape[0]):
            path.lineTo(area_x[i], area_y[i])
        x2 = self.__getLineIntersection(self.arc[idx2-1], self.arc[idx2], 2.0 * self.rad[idx2-1], 2.0 * self.rad[idx2], self.threshold)
        path.lineTo(x2, self.threshold)
        self.stenosis_area = StenosisAreaItem(self.stenosisAreaHoverEnter,
                                              self.stenosisAreaHoverLeave,
                                              pg.mkBrush(STENOSIS_COLORS_QT[colorID]),
                                              pg.mkBrush(STENOSIS_COLORS_QT_H[colorID]),
                                              path)
        self.lineplot.addItem(self.stenosis_area)

        # calculate stenosis degree and information
        min_idx = idx1 + np.argmin(self.rad[idx1:idx2])
        self.nascet_min_dia = 2.0 * self.rad[min_idx]
        self.min_diameter_pos = self.pos[min_idx]
        self.min_diameter_normal = np.mean(self.pos[min_idx:min_idx+6] - self.pos[min_idx-5:min_idx+1], axis=0)
        self.min_diameter_normal /= np.linalg.norm(self.min_diameter_normal)
        half_stenosis_length = int((idx2 - idx1)/2)
        self.stenosis_arc_len = self.arc[idx2] - self.arc[idx1]
        ref_idx = min(idx2 + half_stenosis_length, self.pos.shape[0]-2)
        self.ref_diameter_pos = self.pos[ref_idx]
        self.ref_diameter_normal = np.mean(self.pos[min_idx:min_idx+6] - self.pos[min_idx-5:min_idx+1], axis=0)
        self.ref_diameter_normal /= np.linalg.norm(self.ref_diameter_normal)
        self.__computeStenosisDegree(ref_idx)

        # create 2D text actor
        self.text_idx = idx1 + half_stenosis_length
        self.text_item = pg.TextItem(self.degree_string, color=Q_COLOR_BLACK, anchor=(0.5, 0))
        self.text_item.setPos(self.arc[self.text_idx], self.nascet_min_dia)
        self.text_item_full = pg.TextItem(self.full_description, color=Q_COLOR_BLACK, anchor=(0.5, 0))
        self.text_item_full.setPos(self.arc[self.text_idx], self.nascet_min_dia)
        self.text_item_full.fill = TEXT_BG_BRUSH
        self.lineplot.addItem(self.text_item) # draw above lines

        # create 3D text actor
        self.text_actor = vtk.vtkBillboardTextActor3D()
        self.text_actor.SetInput(self.degree_string)
        self.text_actor.SetDisplayOffset(-20, -5)
        self.text_actor.GetTextProperty().SetFontSize(20)
        self.text_actor.GetTextProperty().SetColor(0, 0, 0)
        self.text_actor.GetTextProperty().SetBackgroundColor(1, 1, 240/255)
        self.vtk_renderer.AddActor(self.text_actor)
        self.update3DTextPos()

        # create 2D nascet reference marker
        self.reference_marker2D = pg.InfiniteLine(pos=self.arc[ref_idx], 
                                                  angle=90,
                                                  movable=True,
                                                  bounds=[self.arc[1], self.arc[-2]],
                                                  pen={'color':STENOSIS_COLORS_QT[colorID], 'width':2})
        self.reference_marker2D.sigDragged.connect(self.referenceMoved)
        self.lineplot.addItem(self.reference_marker2D)

        # create 3D nascet reference marker
        self.reference_marker3D = vtk.vtkLineSource()
        self.reference_marker3D.SetPoint1(self.pos[ref_idx-1])
        self.reference_marker3D.SetPoint2(self.pos[ref_idx+1])
        self.tube_filter = vtk.vtkTubeFilter()
        self.tube_filter.SetInputConnection(self.reference_marker3D.GetOutputPort())
        self.tube_filter.SetRadius(1.5*self.rad[ref_idx])
        self.tube_filter.SetNumberOfSides(25)
        self.tube_filter.CappingOn()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.tube_filter.GetOutputPort())
        self.reference_actor = vtk.vtkActor()
        self.reference_actor.SetMapper(mapper)
        self.reference_actor.GetProperty().SetColor(STENOSIS_COLORS_VTK[colorID])
        self.vtk_renderer.AddActor(self.reference_actor)


    def __getLineIntersection(self, x1, x2, y1, y2, h):
        """
        Computes the x-value of the intersection of a line
        through the points (x1, y1), (x2, y2) with the horizontal
        line at y=h.
        Used to display the correct 2D stenosis area.
        """
        m = (y2 - y1) / (x2 - x1)
        b = y2 - m * x2
        return (h - b) / m

        
    def __computeStenosisDegree(self, ref_idx):
        nascet_ref_dia = 2.0 * self.rad[ref_idx]
        self.degree = ((nascet_ref_dia - self.nascet_min_dia) / nascet_ref_dia) * 100.0
        self.degree_string = f'{self.degree:.1f}%'
        self.full_description =  f'Stenosis degree (NASCET): {self.degree_string}\n'\
                                 f'Smallest inner diameter: {self.nascet_min_dia:.1f} mm\n'\
                                 f'Reference diameter: {nascet_ref_dia:.1f} mm\n'\
                                 f'Stenosis length: {self.stenosis_arc_len:.1f} mm'

    
    def removeUnconnectedGeometry(self):
        connectivityFilter = vtk.vtkConnectivityFilter()
        connectivityFilter.SetInputConnection(self.clipper.GetOutputPort())
        connectivityFilter.SetExtractionModeToLargestRegion()
        connectivityFilter.ColorRegionsOff()
        self.mapper.SetInputConnection(connectivityFilter.GetOutputPort())


    def update3DTextPos(self):
        cam_pos = np.array(self.vtk_renderer.GetActiveCamera().GetPosition())
        view_dir = self.pos[self.text_idx] - cam_pos
        text_pos = cam_pos + 0.7 * view_dir
        self.text_actor.SetPosition(text_pos)

    def referenceMoved(self):
        # re-compute stenosis degree, update description
        idx = np.searchsorted(self.arc, self.reference_marker2D.getXPos())
        self.ref_diameter_pos = self.pos[idx]
        self.ref_diameter_normal = np.mean(self.pos[idx:idx+6] - self.pos[idx-5:idx+1], axis=0)
        self.ref_diameter_normal /= np.linalg.norm(self.ref_diameter_normal)
        self.__computeStenosisDegree(idx)

        # update scene
        self.text_actor.SetInput(self.degree_string)
        self.text_item.setPlainText(self.degree_string)
        self.text_item_full.setPlainText(self.full_description)
        self.reference_marker3D.SetPoint1(self.pos[idx-1])
        self.reference_marker3D.SetPoint2(self.pos[idx+1])
        self.tube_filter.SetRadius(1.5*self.rad[idx])
        self.vtk_renderer.GetRenderWindow().Render()


    def stenosisAreaHoverEnter(self):
        self.text_actor.SetInput(self.full_description)
        self.text_actor.SetDisplayOffset(-120, -25)
        self.text_actor.GetTextProperty().SetBackgroundOpacity(1)
        self.lineplot.addItem(self.text_item_full)
        self.stenosis_actor.GetProperty().SetColor(STENOSIS_COLORS_VTK_H[self.colorID])
        self.reference_actor.GetProperty().SetColor(STENOSIS_COLORS_VTK_H[self.colorID])
        self.update3DTextPos()
        self.vtk_renderer.GetRenderWindow().Render()

    
    def stenosisAreaHoverLeave(self):
        self.text_actor.SetInput(self.degree_string)
        self.text_actor.SetDisplayOffset(-20, -5)
        self.text_actor.GetTextProperty().SetBackgroundOpacity(0)
        self.lineplot.removeItem(self.text_item_full)
        self.stenosis_actor.GetProperty().SetColor(STENOSIS_COLORS_VTK[self.colorID])
        self.reference_actor.GetProperty().SetColor(STENOSIS_COLORS_VTK[self.colorID])
        self.vtk_renderer.GetRenderWindow().Render()


    def cleanup(self):
        self.vtk_renderer.RemoveActor(self.stenosis_actor)
        self.vtk_renderer.RemoveActor(self.text_actor)
        self.vtk_renderer.RemoveActor(self.reference_actor)
        self.lineplot.removeItem(self.stenosis_area)
        self.lineplot.removeItem(self.text_item)
        self.lineplot.removeItem(self.text_item_full)
        self.lineplot.removeItem(self.reference_marker2D)



class StenosisClassifierTab(QWidget):
    """
    Tab view of a right OR left side carotid for stenosis classification.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        # create shortcut for saving scene meta information
        self.save_filename = ""
        self.save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self.save_shortcut.activated.connect(self.save_scene)

        self.min_branch_len = 20 # minimal length of a branch in mm
        self.branch_cutoff = 1  # length to be cut from branch ends in mm

        self.c_radii_lists = []     # processed centerline radii
        self.c_pos_lists = []       # processed centerline positions
        self.c_arc_lists = []       # processed centerline arc length (cumulated)
        self.c_parent_indices = []  # index tuples for branch parent / branch point
        self.c_stenosis_lists = []  # lists of stenosis objects per centerline
        self.nr_stenoses = 0        # total number of displayed stenoses

        # model view
        interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        self.model_view = QVTKRenderWindowInteractor(self)
        self.model_view.SetInteractorStyle(interactor_style)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1,1,1)
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(0, 0, -100)
        cam.SetFocalPoint(0, 0, 0)
        cam.SetViewUp(0, -1, 0)
        cam.AddObserver("ModifiedEvent", self.cameraModifiedEvent)
        self.model_view.GetRenderWindow().AddRenderer(self.renderer)

        # graph view
        self.widget_lineplots = pg.GraphicsLayoutWidget()
        self.lineplots = []

        # combine all in a layout
        self.top_layout = QHBoxLayout(self)
        self.top_layout.addWidget(self.widget_lineplots)
        self.top_layout.addWidget(self.model_view)

        # lumen display vtk pipeline
        self.reader_lumen = vtk.vtkSTLReader()
        normals = vtk.vtkTriangleMeshPointNormals()
        normals.SetInputConnection(self.reader_lumen.GetOutputPort())
        shrink_layer0 = vtk.vtkWarpVector()
        shrink_layer0.SetInputConnection(normals.GetOutputPort())
        shrink_layer0.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, vtk.vtkDataSetAttributes.NORMALS)
        shrink_layer0.SetScaleFactor(-0.02) # shrink lumen by small factor to resolve stenosis geometry overlaps
        self.mapper_lumen = vtk.vtkPolyDataMapper()
        self.mapper_lumen.SetInputConnection(shrink_layer0.GetOutputPort())
        self.actor_lumen = vtk.vtkActor()
        self.actor_lumen.SetMapper(self.mapper_lumen)
        self.actor_lumen.GetProperty().SetColor(1,1,1)

        # branch display vtk pipeline
        self.reader_centerline = vtk.vtkXMLPolyDataReader()
        self.shrink_layer1 = vtk.vtkWarpVector()
        self.shrink_layer1.SetInputConnection(normals.GetOutputPort())
        self.shrink_layer1.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, vtk.vtkDataSetAttributes.NORMALS)
        self.shrink_layer1.SetScaleFactor(-0.01) # shrink lumen by small factor to resolve stenosis geometry overlaps
        self.branch_actors = []

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

    def save_scene(self):
        if len(self.save_filename) == 0 or len(self.c_stenosis_lists[0]) == 0:
            print("Nothing to save.")
            return

        # collect meta information on the highest ACI stenosis
        ACI_stenoses = self.c_stenosis_lists[0]
        stenosis = ACI_stenoses[0]
        for s in ACI_stenoses:
            if s.degree > stenosis.degree:
                stenosis = s
        d = {"diameter_threshold": stenosis.threshold,
             "stenosis_degree":float(stenosis.degree),
             "stenosis_length":float(stenosis.stenosis_arc_len),
             "stenosis_min_p":stenosis.min_diameter_pos.tolist(),
             "stenosis_min_n":stenosis.min_diameter_normal.tolist(),
             "poststenotic_arcval":float(stenosis.reference_marker2D.getXPos()),
             "poststenotic_p":stenosis.ref_diameter_pos.tolist(),
             "poststenotic_n":stenosis.ref_diameter_normal.tolist()}

        # save dict to json
        try:
            with open(self.save_filename, 'w') as f:
                json.dump(d, f)
            print("Saved " + self.save_filename)
        except: 
            print("Could not write file.")

    
    def cameraModifiedEvent(self, obj, ev):
        for stenosis_list in self.c_stenosis_lists:
            for stenosis in stenosis_list:
                stenosis.update3DTextPos()
    

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
            self.__preprocessCenterlines()
            self.plot_radii()

            # load scene if saved
            self.save_filename = lumen_file[:-9] + "meta.txt"
            # if os.path.exists(self.save_filename):
            #     try:
            #         with open(self.save_filename, 'r') as f:
            #             d = json.load(f)
            #     except:
            #         print("Classifier module could not load " + self.save_filename)
            # TODO set scene

        else:
            # clear all
            self.clearStenoses()
            self.widget_lineplots.clear()
            self.lineplots = []
            self.renderer.RemoveActor(self.actor_lumen)
            for actor in self.branch_actors:
                self.renderer.RemoveActor(actor)
            self.branch_actors = []
            self.text_patient.SetInput("No lumen or centerlines file found for this side.")
            self.save_filename = ""

        # reset scene and render
        self.renderer.ResetCamera()
        self.model_view.GetRenderWindow().Render()


    def __preprocessCenterlines(self):
        # lists for each line in centerlines
        # lines are ordered source->outlet
        self.c_pos_lists = []       # 3xn numpy arrays with point positions
        self.c_arc_lists = []       # 1xn numpy arrays with arc length along centerline (accumulated)
        self.c_radii_lists = []     # 1xn numpy arrays with maximal inscribed sphere radius
        self.c_parent_indices = []  # tuple per list: (parent idx, branch point idx)
        self.clearStenoses() # empties the list of actors and removes them from rendering

        # iterate all (global) lines
        # each line is a vtkIdList containing point ids in the right order
        centerlines = self.reader_centerline.GetOutput()
        radii_flat = centerlines.GetPointData().GetArray('MaximumInscribedSphereRadius')
        l = centerlines.GetLines()
        l.InitTraversal()
        for i in range(l.GetNumberOfCells()):
            pointIds = vtk.vtkIdList()
            l.GetNextCell(pointIds)

            # retrieve position data
            points = vtk.vtkPoints()
            centerlines.GetPoints().GetPoints(pointIds, points)
            p = vtk_to_numpy(points.GetData())

            # calculate arc len
            arc = p - np.roll(p, 1, axis=0)
            arc = np.sqrt((arc*arc).sum(axis=1))
            arc[0] = 0
            arc = np.cumsum(arc)

            # retrieve radius data
            radii = vtk.vtkDoubleArray()
            radii.SetNumberOfTuples(pointIds.GetNumberOfIds())
            radii_flat.GetTuples(pointIds, radii)
            r = vtk_to_numpy(radii)

            # add to centerlines
            self.c_pos_lists.append(p)
            self.c_arc_lists.append(arc)
            self.c_radii_lists.append(r)
            self.c_parent_indices.append((i,0)) # points to own origin
            self.c_stenosis_lists.append([]) # for storing actors later

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

        # create branch clippers
        self.branch_actors = []
        start_idx = 0
        for i in range(len(self.c_radii_lists)):
            branch_clip_function = vtk.vtkImplicitBoolean()
            branch_clip_function.SetOperationTypeToUnion()
            radii2 = 2.0*self.c_radii_lists[i]
            pos = self.c_pos_lists[i]
            for idx in range(start_idx, radii2.shape[0], 20):
                sphere = vtk.vtkSphere()
                sphere.SetCenter(pos[idx])
                sphere.SetRadius(radii2[idx])
                branch_clip_function.AddFunction(sphere)
            sphere = vtk.vtkSphere()
            sphere.SetCenter(pos[radii2.shape[0]-1])
            sphere.SetRadius(radii2[radii2.shape[0]-1])
            branch_clip_function.AddFunction(sphere)
            branch_clipper = vtk.vtkClipPolyData()
            branch_clipper.SetInputConnection(self.shrink_layer1.GetOutputPort())
            branch_clipper.SetClipFunction(branch_clip_function)
            branch_clipper.SetInsideOut(True)
            branch_mapper = vtk.vtkPolyDataMapper()
            branch_mapper.SetInputConnection(branch_clipper.GetOutputPort())
            branch_actor = vtk.vtkActor()
            branch_actor.SetMapper(branch_mapper)
            branch_actor.GetProperty().SetColor(1.0, 1.0, 0.2)
            self.branch_actors.append(branch_actor)
            start_idx = 80 # start subbranches later

    
    def linePlotHoverEnter(self, lineplot):
        self.diameter_plots[lineplot.plotID].setPen(LINEPLOT_WIDE_PEN)
        self.renderer.AddActor(self.branch_actors[lineplot.plotID])
        self.renderer.GetRenderWindow().Render()

    
    def linePlotHoverLeave(self, lineplot):
        self.diameter_plots[lineplot.plotID].setPen(LINEPLOT_DEFAULT_PEN)
        self.renderer.RemoveActor(self.branch_actors[lineplot.plotID])
        self.renderer.GetRenderWindow().Render()


    def plot_radii(self):
        self.widget_lineplots.clear()
        self.lineplots = []
        self.diameter_plots = []
        dashed_pen = pg.mkPen({'color':(0,0,0), 'width':0.5, 'style':Qt.DashLine})
        
        for i in range(len(self.c_radii_lists)):
            lineplot = HoverablePlotItem(i)
            lineplot.sigHoverEnter[object].connect(self.linePlotHoverEnter)
            lineplot.sigHoverLeave[object].connect(self.linePlotHoverLeave)
            lineplot.setLabel('left', "Minimal Diameter (mm)")
            lineplot.showGrid(x=False, y=True, alpha=0.2)
            self.widget_lineplots.addItem(lineplot)
            self.lineplots.append(lineplot)

            # draw diemeter lineplot
            d_plot = lineplot.plot(x=self.c_arc_lists[i], y=2.0*self.c_radii_lists[i], pen=LINEPLOT_DEFAULT_PEN)
            self.diameter_plots.append(d_plot)
            lineplot.disableAutoRange()

            # mark origin of subbranches
            subbranch_ids = []
            for j, index_tuple in enumerate(self.c_parent_indices):
                if index_tuple[0] == i and index_tuple[1] != 0:
                    subbranch_ids.append(index_tuple[1])
                    x = self.c_arc_lists[i][index_tuple[1]]
                    lineplot.addItem(pg.InfiniteLine(pos=x, angle=90, pen=dashed_pen))
            subbranch_ids = sorted(subbranch_ids)

            # draw horizontal sliders
            subbranch_ids.insert(0, 0) # line start id
            subbranch_ids.append(len(self.c_arc_lists[i])-1) # line end id
            for j in range(len(subbranch_ids)-1):
                id0 = subbranch_ids[j]
                id1 = subbranch_ids[j+1]
                x_min = self.c_arc_lists[i][id0]
                x_max = self.c_arc_lists[i][id1]
                max_y = 2.0 * np.max(self.c_radii_lists[i][id0:id1])
                min_y = 2.0 * np.min(self.c_radii_lists[i][id0:id1])
                min_y -= 0.01 * (max_y-min_y) # small offset below lowest value
                selection_line = LineROI(i, x_min, x_max, y_pos=min_y, y_bounds=[min_y, max_y],
                                         angle=0, movable=True, pen={'color':(0,0,0), 'width':1})
                lineplot.addItem(selection_line)
                selection_line.sigPositionChanged[object].connect(self.lineROIposChanged)
                selection_line.sigPositionChangeFinished[object].connect(self.lineROIPosChangeFinished)

            # next graph
            self.widget_lineplots.nextRow()

        # link axes, set view ranges
        max_x = self.c_arc_lists[0][-1]
        min_y = 2.0 * np.min(self.c_radii_lists[0])
        max_y = 2.0 * np.max(self.c_radii_lists[0])
        self.lineplots[0].setYRange(min_y, max_y, padding=0.15)
        for i in range(1, len(self.lineplots)):
            self.lineplots[i].setXLink(self.lineplots[0])
            min_y = 2.0 * np.min(self.c_radii_lists[i])
            max_y = 2.0 * np.max(self.c_radii_lists[i])
            self.lineplots[i].setYRange(min_y, max_y, padding=0.15)
            if self.c_arc_lists[i][-1] > max_x:
                max_x = self.c_arc_lists[i][-1]
        self.lineplots[0].setXRange(0, max_x)

        # set label
        self.lineplots[-1].setLabel('bottom', "Branch Length (mm)")

    
    def clearStenoses(self):
        for stenosis_list in self.c_stenosis_lists:
            for stenosis in stenosis_list:
                stenosis.cleanup()
        self.c_stenosis_lists.clear()
        self.nr_stenoses = 0


    def lineROIposChanged(self, lineROI):
        pos = self.c_pos_lists[lineROI.plot_id]
        rad = self.c_radii_lists[lineROI.plot_id]
        arc = self.c_arc_lists[lineROI.plot_id]
        stenosis_list = self.c_stenosis_lists[lineROI.plot_id]
        r_thresh = lineROI.getYPos() / 2.0

        start_index = np.searchsorted(arc, lineROI.x_start)
        end_index = np.searchsorted(arc, lineROI.x_end) + 1
        radii_ranges = np.where(rad[start_index:end_index] < r_thresh, 0, 1)
        radii_ranges[-1] = 1 # closes open ends
        radii_ranges = radii_ranges - np.roll(radii_ranges, 1)

        # compute threshold indices, save in lineROI for later use
        indices_down = np.where(radii_ranges == -1)[0] + start_index
        indices_up = np.where(radii_ranges == 1)[0] + start_index
        nr_stenoses = indices_down.size
        assert nr_stenoses == indices_up.size

        # cleanup all stenoses with same start index
        for i in range(len(stenosis_list)-1, -1, -1):
            s = stenosis_list[i]
            if s.start_index == start_index:
                s.cleanup()
                del stenosis_list[i]
                self.nr_stenoses -= 1

        # create a stenosis object for each case
        for i in range(indices_down.size):
            idx1 = indices_down[i]
            idx2 = indices_up[i]
            
            # catch if too close to branch end
            if (idx1 <= 10 or 
                idx2 >= pos.shape[0] - 10 or
                idx1 == start_index or
                idx2 == end_index - 1):
                continue

            stenosis = StenosisWrapper(self.renderer, 
                                       self.lineplots[lineROI.plot_id],
                                       lineROI,
                                       self.reader_lumen.GetOutput(),
                                       pos,
                                       rad,
                                       arc,
                                       start_index,
                                       idx1,
                                       idx2,
                                       colorID=self.nr_stenoses%len(STENOSIS_COLORS))
            stenosis_list.append(stenosis)
            self.nr_stenoses += 1
            stenosis.update3DTextPos()

        self.model_view.GetRenderWindow().Render()


    def lineROIPosChangeFinished(self, lineROI):
        for stenosis in self.c_stenosis_lists[lineROI.plot_id]:
            stenosis.removeUnconnectedGeometry()
        self.model_view.GetRenderWindow().Render()


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