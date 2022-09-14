#!/usr/bin/env python3
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtGui import QColor, QImage, QPixmap
from PySide2.QtWidgets import  QColorDialog, QGraphicsPixmapItem, QGraphicsScene#QTreeWidgetItem
import pyqtgraph
import os
import sys
import numpy as np
import threading
import pathlib
import tifffile
import datetime
from PIL import Image
import scipy.ndimage
import matplotlib._contour as contour
import threading
import queue
from PySide2.QtUiTools import QUiLoader


SATURATION_R = 45

def dragEnterEvent(self, event):
    event.accept()
def dragMoveEvent(self, event):
    event.accept()
def dragLeaveEvent(self, event):
    event.accept()
#setattr(QGraphicsScene,'dragEnterEvent',dragEnterEvent)
#setattr(QGraphicsScene,'dragMoveEvent',dragMoveEvent)
#setattr(QGraphicsScene,'dragLeaveEvent',dragLeaveEvent)

QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
class UiLoader(QUiLoader):
    def __init__(self, base_instance):
        QUiLoader.__init__(self, base_instance)
        self.base_instance = base_instance

    def createWidget(self, class_name, parent=None, name=''):
        if parent is None and self.base_instance:
            return self.base_instance
        elif class_name == "ImageView":
            return pyqtgraph.ImageView(parent=parent)
        else:
            # create a new widget for child widgets
            widget = QUiLoader.createWidget(self, class_name, parent, name)
            if self.base_instance:
                setattr(self.base_instance, name, widget)
            return widget

def apply_sat_limit(Jab, limit_sat = 'shared'):
    '''
    apply a saturation limit to Jab in order to ensure valid saturation when the limit of the RGB colorspace is reached
    Args:
        Jab: np array of shape (n,m,3) encoded in the colorspace
        limit_sat: 'shared' or 'individual'
            if 'shared', all hues share same limit to saturation (the minimum where all saturation values present in the colormap can be represented)
            if 'individual', different hues have different sauration limits
    returns:
        None (Jab is modified in-place)
    '''

    #limit = sat[valid_argmax]
    #limit_ax_0_J = J
    #limit_ax_1_phi = phi
    limit, limit_ax_0_J, limit_ax_1_phi = get_sat_limts()
    inerpolator = scipy.interpolate.RectBivariateSpline(limit_ax_0_J, limit_ax_1_phi, limit)

    phi = np.arctan2(Jab[:,:,1],Jab[:,:,2])
    sat = np.sqrt(Jab[:,:,1]**2 + Jab[:,:,2]**2)

    max_sat = inerpolator( Jab[:,:,0], phi, grid = False)
    if limit_sat == 'shared':
        max_sat[:,:] = np.min(max_sat, axis=1)[:,np.newaxis]
    mask = sat>max_sat
    #sat[mask] = max_sat[mask]
    change = (max_sat[mask]+0.000000001)/(sat[mask]+0.000000001)
    Jab[mask,1] *= change
    Jab[mask,2] *= change

def get_sat_limts():
    '''
        returns the a 2d matrix of approximate limits to sat (radius in a-b space) in terms of phi and J
    '''

    if not 'limit' in globals():
        global limit, limit_ax_0_J, limit_ax_1_phi

        phi = np.linspace(-np.pi, np.pi, 256+1)
        J = np.linspace(12.5, 129.5,128)
        sat = np.linspace(0,70,256)

        J_phi_sat = np.empty((len(J),len(phi),len(sat),3))
        J_phi_sat[:,:,:,0] = J[:,np.newaxis,np.newaxis]
        J_phi_sat[:,:,:,1] = phi[np.newaxis,:,np.newaxis]
        J_phi_sat[:,:,:,2] = sat[np.newaxis,np.newaxis,:]

        Jab = np.empty(J_phi_sat.shape)
        Jab[:,:,:,0] = J_phi_sat[:,:,:,0]
        Jab[:,:,:,1] = J_phi_sat[:,:,:,2]*np.sin(J_phi_sat[:,:,:,1])
        Jab[:,:,:,2] = J_phi_sat[:,:,:,2]*np.cos(J_phi_sat[:,:,:,1])
        rgb = colorspacious.cspace_convert(Jab, 'CAM02-LCD', 'sRGB255')
        rgb[rgb>255.5] = np.nan
        rgb[rgb<-0.5] = np.nan

        flat_rgb = np.sum(rgb, axis = -1)
        flat_rgb[:,:,0] = 0
        ''' no need for this here!
        # there are some strange regsions in the limits-overview because there are 'jumps' as we go through phi
        # therefore limit the derivative in phi
        for i, _ in enumerate(sat[:-1]):
            flat_rgb[:,0,i]  +=  flat_rgb[:,-1,i]
            flat_rgb[:,-1,i] +=  flat_rgb[:,0,i]
            flat_rgb[:,1:,i+1]  +=  flat_rgb[:,:-1,i]
            flat_rgb[:,:-1,i+1] +=  flat_rgb[:,1:,i]

        flat_rgb[:,0,-1]  +=  flat_rgb[:,-1,-1]
        flat_rgb[:,-1,-1] +=  flat_rgb[:,0,-1]
        '''
        valid = np.invert(np.isnan(flat_rgb)) + np.linspace(0,0.9,len(sat))[np.newaxis,np.newaxis,:]
        valid_argmax = np.argmax(valid, axis = -1)

        limit = sat[valid_argmax]
        limit_ax_0_J = J
        limit_ax_1_phi = phi
    return limit, limit_ax_0_J, limit_ax_1_phi


def apply_J_limit(Jab, limiting_type = 'shared'):
    '''
    apply a lightness limit to Jab in order to ensure valid saturation when the limit of the RGB colorspace is reached
    Args:
        Jab: np array of shape (n,m,3) encoded in the colorspace
        limit_J: 'shared' or 'individual'
            if 'shared', all hues share same limit to J (the minimum where all J values present in the colormap can be represented)
            if 'individual', different hues have different J limits
    returns:
        None (Jab is modified in-place)
    '''

    #limit = sat[valid_argmax]
    #limit_ax_0_J = J
    #limit_ax_1_phi = phi
    limit, limit_ax_2_sat, limit_ax_1_phi = get_J_limts()
    inerpolator = scipy.interpolate.RectBivariateSpline(limit_ax_1_phi, limit_ax_2_sat, limit)

    phi = np.arctan2(Jab[:,:,1],Jab[:,:,2])
    sat = np.sqrt(Jab[:,:,1]**2 + Jab[:,:,2]**2)

    max_J = inerpolator( phi, sat, grid = False)
    if limiting_type == 'shared':
        max_J[:,:] = np.max(max_J, axis=1)[:,np.newaxis]
    mask = Jab[:,:,0]>max_J
    #sat[mask] = max_J[mask]
    #   change = (max_J[mask]+0.000000001)/(Jab[mask,0]+0.000000001)
    Jab[mask,0] = max_J[mask]



def get_J_limts():
    '''
        returns the a 2d matrix of approximate limits to sat (radius in a-b space) in terms of phi and J
    '''

    if not 'J_limit' in globals():
        global J_limit, J_limit_ax_2_sat, J_limit_ax_1_phi

        phi = np.linspace(-np.pi, np.pi, 256+1)
        J = np.linspace(12.5, 129.5,128)
        sat = np.linspace(0,70,256)

        J_phi_sat = np.empty((len(J),len(phi),len(sat),3))
        J_phi_sat[:,:,:,0] = J[:,np.newaxis,np.newaxis]
        J_phi_sat[:,:,:,1] = phi[np.newaxis,:,np.newaxis]
        J_phi_sat[:,:,:,2] = sat[np.newaxis,np.newaxis,:]

        Jab = np.empty(J_phi_sat.shape)
        Jab[:,:,:,0] = J_phi_sat[:,:,:,0]
        Jab[:,:,:,1] = J_phi_sat[:,:,:,2]*np.sin(J_phi_sat[:,:,:,1])
        Jab[:,:,:,2] = J_phi_sat[:,:,:,2]*np.cos(J_phi_sat[:,:,:,1])
        rgb = colorspacious.cspace_convert(Jab, 'CAM02-LCD', 'sRGB255')
        rgb[rgb>255.5] = np.nan
        rgb[rgb<-0.5] = np.nan

        flat_rgb = np.sum(rgb, axis = -1)
        #flat_rgb[:,:,0] = 0

        # there are some strange regsions in the limits-overview because there are 'jumps' as we go through phi
        # therefore limit the derivative in phi
        '''
        for i, _ in enumerate(sat[:-1]):
            flat_rgb[:,0,i]  +=  flat_rgb[:,-1,i]
            flat_rgb[:,-1,i] +=  flat_rgb[:,0,i]
            flat_rgb[:,1:,i+1]  +=  flat_rgb[:,:-1,i]
            flat_rgb[:,:-1,i+1] +=  flat_rgb[:,1:,i]

        flat_rgb[:,0,-1]  +=  flat_rgb[:,-1,-1]
        flat_rgb[:,-1,-1] +=  flat_rgb[:,0,-1]
        '''
        valid = np.invert(np.isnan(flat_rgb)) + np.linspace(0,0.9,len(J))[:,np.newaxis,np.newaxis]
        valid_argmax = np.argmax(valid, axis = 0)
        valid_argmax[np.max(valid, axis = 0)<1] = 127
        J_limit = J[valid_argmax]
        J_limit_ax_2_sat = sat
        J_limit_ax_1_phi = phi
    return J_limit, J_limit_ax_2_sat, J_limit_ax_1_phi



def set_transform(source, target):
    horz_scroll = source.horizontalScrollBar().value()
    vert_scroll = source.verticalScrollBar().value()
    transform = source.transform()
    #zoom = source._zoom
    # temporary block signals from scroll bars to prevent interference
    #horz_blocked = target.horizontalScrollBar().blockSignals(True)
    #vert_blocked = target.verticalScrollBar().blockSignals(True)
    #target._zoom = zoom
    target.setTransform(transform)
    #dx = horz_scroll - target.horizontalScrollBar().value()
    #dy = vert_scroll - target.verticalScrollBar().value()
    #target.horizontalScrollBar().setValue(dx)
    #target.verticalScrollBar().setValue(dy)
    target.horizontalScrollBar().setValue(horz_scroll)
    target.verticalScrollBar().setValue(vert_scroll)
    #target.horizontalScrollBar().blockSignals(horz_blocked)
    #target.verticalScrollBar().blockSignals(vert_blocked)

def make_update(main_window, Jab, pos, size, angle, jlim, ang, limit_s, curve_levels):
    Jab = np.copy(Jab) #colorspacious.cspace_convert(self.image[:,:,:3], 'sRGB255', 'CAM02-LCD')

    c = np.cos(angle*np.pi/180)
    s = np.sin(angle*np.pi/180)
    center = [pos[0] +size[0]/2*c -size[1]/2*s,
              pos[1] +size[0]/2*s +size[1]/2*c]
    J_ROI_0 = (12.5, 129.5)
    Jab[:,:,0] -= J_ROI_0[0]
    Jab[:,:,0] *= (J_ROI_0[1]-J_ROI_0[0])/(jlim[1]-jlim[0])
    Jab[:,:,0] += jlim[0]

    # scale
    Jab[:,:,1] *= size[1]/256
    Jab[:,:,2] *= size[0]/256
    # rotate
    a          = c*Jab[:,:,1]+s*Jab[:,:,2]
    Jab[:,:,2] = -s*Jab[:,:,1]+c*Jab[:,:,2]
    Jab[:,:,1] = a
    # shift center
    Jab[:,:,1] += (center[1]-128)*SATURATION_R/128
    Jab[:,:,2] += (center[0]-128)*SATURATION_R/128

    rot(Jab, ang)
    apply_sat_limit(limited_Jab := np.copy(Jab),'individual')
    apply_J_limit(Jab,'individual')
    Jab = (1-limit_s)*Jab + limit_s*limited_Jab
    shifted_image = colorspacious.cspace_convert(Jab, 'CAM02-LCD', 'sRGB255')
    shifted_image[shifted_image>255] = 255
    shifted_image[shifted_image<0] = 0

    return shifted_image, make_curves(Jab, ang, curve_levels)[0]

def make_curves(Jab, ang, levels = None):
    a_1 = np.linspace(-SATURATION_R, SATURATION_R,256)
    d2 = 0.5*(a_1[1]-a_1[0])
    range = [[a_1[0]-d2, a_1[-1]-d2], [a_1[0]-d2, a_1[-1]-d2]]
    hist, xedges, yedges = np.histogram2d(Jab[:,:,1].ravel(), Jab[:,:,2].ravel(),
            bins=256, range=range, normed=None, weights=None, density=None)
    hist = scipy.ndimage.gaussian_filter(hist, 1)

    if levels == None:
        levels = [np.percentile(hist,85),np.percentile(hist,93), np.percentile(hist,97),np.percentile(hist,99),np.percentile(hist,99.5),np.percentile(hist,99.75)]
    data = hist.T
    mask, corner_mask, nchunk = None, True, 0
    xx, yy = np.meshgrid(np.arange(256), np.arange(256))
    # may need to change to  contourpy in the future
    qcg = contour.QuadContourGenerator(xx, yy, data, mask, corner_mask, nchunk)
    curves = []
    for i, v in enumerate(levels):
        ## generate isocurve with automatic color selection
        v = np.max([v,1])
        reses = qcg.create_contour(v)
        # there are different versions of matplotlib that give different return types, do a type check to account for this
        if type(reses[0]) != np.ndarray:
            reses = reses[0]
        for res in reses:
            res -= 128
            s = np.sin(-ang*2*np.pi/360)
            c = np.cos(-ang*2*np.pi/360)
            x        = res[:,0]*c - res[:,1]*s
            res[:,1] = res[:,0]*s + res[:,1]*c
            res[:,0] = x
            res += 128
            #c = pyqtgraph.IsocurveItem(data =data,  level=v, pen=[0,0,0])
            #c.setZValue(100)
            curves.append(res)
    return curves, levels

class UpdateComputer(threading.Thread):
    def __init__(self, main_window, work_queue):
        self.main_window = main_window
        self.work_queue = work_queue
        threading.Thread.__init__(self)
    def run(self):
        while True:
            work = self.work_queue.get()
            if work[0] == 'end':
                return
            while not self.work_queue.empty():
                work = self.work_queue.get()
                if work[0] == 'end':
                    return
            if work[0] == 'update':
                pos, size, angle, jlim, ang, limit_s = work[1:]
                main_window.shifted_image, main_window.new_curves = make_update(self, self.main_window.Jab, pos, size, angle, jlim, ang, limit_s, self.main_window.curve_levels)
                #main_window.shifted_image = shifted_image
                #main_window.new_curves
                self.main_window.rimt(main_window.apply_update)

def rot( Jab, ang):
    #phi = np.arctan2(Jab[:,:,1], Jab[:,:,2])+ang
    #sat = np.sqrt(Jab[:,:,1]**2 + Jab[:,:,2]**2)
    s = np.sin(ang*2*np.pi/360)
    c = np.cos(ang*2*np.pi/360)
    a = Jab[:,:,1]*c - Jab[:,:,2]*s
    Jab[:,:,2] = Jab[:,:,1]*s + Jab[:,:,2]*c
    Jab[:,:,1] = a


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        pyqtgraph.setConfigOption('background', 'w')
        #set theme:
        if getattr(sys, 'frozen', False):
            application_path = os.path.dirname(sys.executable)
        elif __file__:
            application_path = os.path.dirname(__file__)
        path = 'set_theme.py'
        if not os.path.exists(path):
            path = str(application_path) + '/set_theme.py'
        if os.path.exists(path):
            with open(path) as f:
                code = compile(f.read(), path, 'exec')
                exec(code, globals(), locals())


        super(MainWindow, self).__init__(*args, **kwargs)
        #Load the UI Page

        if getattr(sys, 'frozen', False):
            application_path = os.path.dirname(sys.executable)
        elif __file__:
            application_path = os.path.dirname(__file__)

        i = 0
        if os.path.exists('colorshifter.ui'):
            path = 'colorshifter.ui'
        else:
            while not os.path.exists(str(application_path) + '/colorshifter.ui'):
                application_path = application_path + '/..'
                i+=1
                if i>10:
                    break
            path = str(application_path) + '/colorshifter.ui'

        loader = UiLoader(self)
        widget = loader.load(path)

        self.setObjectName("MainWindow")
        # set icon
        self.setWindowIcon(QtGui.QIcon('iconOri.ico'))


        for img in  [self.original_colorspace]:
            img.ui.roiBtn.hide()
            img.ui.menuBtn.hide()
            img.has_img = False
            img.ui.histogram.hide()

        #self.img_0.getHistogramWidget().sigLevelsChanged.connect(self.update_composite_slot)
        self.updating_colors = False




        self.scene_in = QGraphicsScene()
        self.scene_out = QGraphicsScene()
        self.gv_in.setScene(self.scene_in)
        self.gv_out.setScene(self.scene_out)
        self.gv_in.viewport().installEventFilter(self)
        self.gv_out.viewport().installEventFilter(self)

        def accept_event(ev):
            ev.accept()
        for scene in [self.scene_in, self.scene_out]:
            scene.dragEnterEvent = accept_event
            scene.dragMoveEvent = accept_event
            scene.dragLeaveEvent = accept_event

        #self.gv_in.dropEvent = self.image_in_drop
        #self.gv_out.dropEvent = self.image_in_drop
        self.gv_in.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)

        #setattr(self.scene_in,'dragEnterEvent',accept_event)
        #setattr(self.scene_in,'dragMoveEvent',accept_event)
        #setattr(self.scene_in,'dragLeaveEvent',accept_event)

        self.hist_wg = pyqtgraph.PlotWidget()
        self.horizontalLayout_2.addWidget(self.hist_wg)
        self.hist_wg.setFixedWidth(100)
        self.hist_pi = self.hist_wg.getPlotItem()
        #self.hist_pi.showAxis('right')
        self.hist_pi.hideAxis('left')
        self.hist_pi.hideAxis('bottom')
        self.hist_pi.getViewBox().invertX(True)
        self.hist_pi.setMouseEnabled(x=False) # Only allow zoom in Y-axis
        #self.hist_plot = self.hist_pi.addPlot()
        y, x = np.histogram([0], bins=np.linspace(12.5, 129.5, 100))
        self.hist_curve = pyqtgraph.PlotCurveItem(-x,np.sqrt(y), stepMode='center', fillLevel=0, brush=(128,128,128,150))
        self.hist_curve.rotate(-90)
        self.hist_pi.addItem(self.hist_curve)
        self.J_ROI_0 = (12.5, 129.5)
        self.J_ROI = pyqtgraph.LinearRegionItem(values=self.J_ROI_0, brush=(128,128,128,75), hoverBrush=(128,128,128,75), orientation='horizontal', movable=True, bounds=None, span=(0, 1), swapMode='sort', clipItem=None)
        self.hist_pi.addItem(self.J_ROI)
        self.J_ROI.sigRegionChanged.connect(self.roi_changed)

        self.limit_slider.valueChanged.connect(self.roi_changed)
        self.downsample.valueChanged.connect(self.new_file)
        self.rotation.valueChanged.connect(self.new_file)

        send_queue, return_queue = queue.Queue(), queue.Queue()
        self.rimt = rimt(send_queue, return_queue).rimt
        self.rimt_executor = RimtExecutor(send_queue, return_queue)
        self.locked = False

        self.work_queue = queue.Queue()
        uc = UpdateComputer(self, self.work_queue)
        uc.start()


        if len(sys.argv)>1:
            file = sys.argv[1]
            self.new_file(file)

        self.save_button.clicked.connect(self.save_clicked)

        self.label.setText(' By Marie Curie fellow Trygve M. R'+chr(int('00E6', 16))+'der. Use at own risk. MIT lisence. https://github.com/trygvrad/colorshifter')

        ### saving
        self.date_today=str(datetime.date.today())
        self.output_int = 1
        while os.path.exists(f'output/{self.date_today}/{self.output_int}'+'.png') or os.path.exists(f'output/{self.date_today}/{self.output_int}'+'.jpg'):
            self.output_int += 1
        self.path.setText(f'output/{self.date_today}/{self.output_int}')
        self.updating_img = False
        self.splitter.setSizes([300,300])

        self.levels = np.array([[0, 255.0], [0, 255], [0, 255]])
        for img in  [self.original_colorspace]:
            img.getImageItem().setLevels(self.levels)
        #colorstamps.stamps.get_sat_limts()
        self.stamp = colorstamps.stamps.get_const_J(J=70, a=(-1, 1), b=(-1, 1), r=SATURATION_R, l=256, mask='no_mask', rot=0)*255

        #self.original_colorspace.getImageItem().setImage(np.transpose(self.stamp, axes = (1,0,2)), autoLevels = False, levelMode = 'rgb')
        # for dragging image
        self.startPos = None

        # roi stuff
        self.roi = self.original_colorspace.roi
        self.roi.show()
        self.roi.setSize((256,256), update=True, finish=False)
        self.roi.sigRegionChanged.connect(self.roi_changed)
        self.roi.removeHandle(0)
        self.roi.addScaleHandle([1, 1], [0.5, 0.5])
        self.roi.invertible = True

        # pic item
        pic = QGraphicsPixmapItem()
        #self.scene.setSceneRect(0, 0, 400, 400)
        self.scene_in.addItem(pic)
        self.in_pic = pic
        self.in_pic.setTransformationMode(QtCore.Qt.SmoothTransformation)

        pic = QGraphicsPixmapItem()
        #self.scene.setSceneRect(0, 0, 400, 400)
        self.scene_out.addItem(pic)
        self.out_pic = pic
        self.out_pic.setTransformationMode(QtCore.Qt.SmoothTransformation)


        #file = '/home/trygvrad/colorshifter/Screenshot from 2022-09-06 11-03-52.png'
        #self.new_file(file)

    def save_clicked(self,event):
        filepath = self.path.text()
        os.makedirs('/'.join(filepath.split('/')[:-1]), exist_ok = True)
        image = np.array(Image.open(self.file))

        Jab = colorspacious.cspace_convert(image[:,:,:3], 'sRGB255', 'CAM02-LCD')
        pos = self.roi.pos()
        size = self.roi.size()
        angle = self.roi.angle()
        jlim = self.J_ROI.getRegion()
        ang = self.rotation.value()
        limit_s = float(self.limit_slider.value())/100
        shifted_image, _ = make_update(self, Jab, pos, size, angle, jlim, ang, limit_s, self.curve_levels)

        image[:,:,:3] = shifted_image
        o_image = Image.fromarray(np.array(np.round(image), dtype = np.uint8))
        o_image.save(filepath+'.'+self.file.split('.')[-1])

        if filepath == f'output/{self.date_today}/{self.output_int}':
            self.output_int += 1
            self.path.setText(f'output/{self.date_today}/{self.output_int}')





    def image_in_drop(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        file = files[0]
        self.new_file(file)



    def eventFilter(self, source, event):
        gv = self.gv_in
        gv2 = self.gv_out
        if source == self.gv_out.viewport():
            gv = self.gv_out
            gv2 = self.gv_in
        if source == gv.viewport() and event.type() == QtCore.QEvent.Drop:
            self.image_in_drop(event)
            return True
        elif source == gv.viewport() and event.type() == QtCore.QEvent.Wheel:
            #if source == gv.viewport() and type(event) == PyQt5.QtGui.QWheelEvent:
            if event.angleDelta().y() > 0:
                scale = 1.125
            else:
                scale = .9

            view_pos = event.pos()
            scene_pos = gv.mapToScene(view_pos)
            gv.centerOn(scene_pos)
            gv.scale(scale, scale)
            delta = gv.mapToScene(view_pos) - gv.mapToScene(gv.viewport().rect().center())
            gv.centerOn(scene_pos - delta)

            set_transform(gv, gv2)
            # do not propagate the event to the scroll area scrollbars
            return True
        elif source == gv.viewport() and event.type() == QtCore.QEvent.MouseButtonPress:
            self.startPos = np.array([event.pos().x(),event.pos().y()], dtype = np.float64)
            self.start_gv = gv
            self.other_gv = gv2
            return True
        elif (self.startPos is not None) and event.type() == QtCore.QEvent.MouseMove:
            # compute the difference between the current cursor position and the
            # previous saved origin point
            delta = self.startPos - np.array([event.pos().x(),event.pos().y()], dtype = np.float64)
            # get the current transformation (which is a matrix that includes the
            # scaling ratios
            transform = self.start_gv.transform()
            # m11 refers to the horizontal scale, m22 to the vertical scale;
            # divide the delta by their corresponding ratio
            deltaX = delta[0] #/ transform.m11()
            deltaY = delta[1] #/ transform.m22()
            # translate the current sceneRect by the delta
            horz_scroll = self.start_gv.horizontalScrollBar().value()
            vert_scroll = self.start_gv.verticalScrollBar().value()
            self.start_gv.horizontalScrollBar().setValue(horz_scroll+int(deltaX))
            self.start_gv.verticalScrollBar().setValue(vert_scroll+int(deltaY))
            self.other_gv.horizontalScrollBar().setValue(horz_scroll+int(deltaX))
            self.other_gv.verticalScrollBar().setValue(vert_scroll+int(deltaY))
            self.startPos += np.array([horz_scroll-self.start_gv.horizontalScrollBar().value(),
                                       vert_scroll-self.start_gv.verticalScrollBar().value()]) #*transform.m11()
            return True
        elif (self.startPos is not None) and event.type() == QtCore.QEvent.MouseButtonRelease:
            self.startPos = None
            return True
        return super().eventFilter(source,event)

    def array_to_QPixmap(self,img):
        w,h,ch = img.shape
        # Convert resulting image to pixmap
        qimg = QImage(img.data, h, w, 3*h, QImage.Format_RGB888)
        qpixmap = QPixmap(qimg)
        return qpixmap


    def new_file(self,file = None):
        if type(file) != type(''):
            if hasattr(self,'file'):
                file = self.file
            else:
                return
        self.file = file
        formats = ['png','jpg','jpeg']
        if file.split('.')[-1] in formats :
            downsample = self.downsample.value()
            image = np.array(Image.open(file))[::downsample, ::downsample,:3]

            self.in_pic.setPixmap(self.array_to_QPixmap(np.array(image)))
            self.out_pic.setPixmap(self.array_to_QPixmap(np.array(image)))
            #self.image_in.getImageItem().setImage(np.transpose(image, axes = (1,0,2)), autoLevels = False, levelMode = 'rgb')
            #self.image_out.getImageItem().setImage(np.transpose(image, axes = (1,0,2)), autoLevels = False, levelMode = 'rgb')

            #v0 = self.image_in.getView()
            #self.image_out.getView().setXLink(v0)
            #self.image_out.getView().setYLink(v0)

            v0 = self.original_colorspace.getView()

            Jab = colorspacious.cspace_convert(image[:,:,:3], 'sRGB255', 'CAM02-LCD')
            ang = self.rotation.value()
            rot(Jab, -ang)
            self.Jab = Jab

            self.stamp = colorstamps.stamps.get_const_J(J=70, a=(-1, 1), b=(-1, 1),
                                            r=SATURATION_R, l=256, mask='no_mask', rot=-ang)*255
            self.original_colorspace.getImageItem().setImage(np.transpose(self.stamp, axes = (1,0,2)), levelMode = 'rgb')


            y, x = np.histogram(Jab[:,:,0].ravel(), bins=np.linspace(12.5, 129.5, 100))
            self.hist_curve.setData(-x,y)
            #curve = pyqtgraph.PlotCurveItem(-x,np.sqrt(y), stepMode=True, fillLevel=0, brush=(128,128,128,150))
            #curve.rotate(-90)
            #self.hist_pi.addItem(curve)

            #J_ROI = pyqtgraph.LinearRegionItem(values=(12.5, 129.5), brush=(128,128,128,75), hoverBrush=(128,128,128,75), orientation='horizontal', movable=True, bounds=None, span=(0, 1), swapMode='sort', clipItem=None)
            #self.hist_pi.addItem(J_ROI)

            #hist = np.log(hist+0.1)

            if hasattr(self, 'ori_curves'):
                for c in self.ori_curves:
                    self.original_colorspace.removeItem(c)


            self.ori_curves = []

            curves, self.curve_levels = make_curves(Jab, ang)
            for res in curves:
                c =  pyqtgraph.PlotDataItem(res[:,1], res[:,0], pen='w')
                c.setParentItem(self.original_colorspace.getImageItem())  ## make sure isocurve is always correctly displayed over image
                self.ori_curves.append(c)
                self.original_colorspace.addItem(c)

            self.update_img()
            #self.roi.mouseDragHandler.mouseDragEvent = self.mouseDragEvent
            #self.roi.mouseDragEvent = self.mouseDragEvent


    def roi_changed(self):
        if self.updating_img == False:
            self.updating_img = True
            QtCore.QTimer.singleShot(250, self.update_img)

    def update_img(self):
        self.updating_img = False
        pos = self.roi.pos()
        size = self.roi.size()
        angle = self.roi.angle()
        jlim = self.J_ROI.getRegion()
        ang = self.rotation.value()

        limit_s = float(self.limit_slider.value())/100
        self.work_queue.put(['update', pos, size, angle, jlim, ang, limit_s])
        #make_update(self, Jab, pos, size, angle, jlim, ang, limit_s)

    def apply_update(self):
        if hasattr(self, 'fit_curves'):
            for c in self.fit_curves:
                self.original_colorspace.removeItem(c)
        self.fit_curves = []
        for res in self.new_curves:
            c =  pyqtgraph.PlotDataItem(res[:,1], res[:,0], pen=[0,0,0])
            c.setParentItem(self.original_colorspace.getImageItem())  ## make sure isocurve is always correctly displayed over image
            self.fit_curves.append(c)
            self.original_colorspace.addItem(c)
        self.out_pic.setPixmap(self.array_to_QPixmap(np.array(np.round(self.shifted_image), dtype = np.uint8)))



    @QtCore.Slot(object)
    def update_composite_slot(self, *args):
        if self.updating_colors == False:
            self.updating_colors = True
            QtCore.QTimer.singleShot(10, self.update_composite)
    def closeEvent(self, event):
        self.work_queue.put(['end'])


import queue
import functools
class rimt():
    def __init__(self, send_queue, return_queue):
        self.send_queue = send_queue
        self.return_queue = return_queue
        self.main_thread = threading.current_thread()

    def rimt(self, function, *args, **kwargs):
        if threading.current_thread() == self.main_thread:
            return function(*args, **kwargs)
        else:
            self.send_queue.put(functools.partial(function, *args, **kwargs))
            return_parameters = self.return_queue.get(True)  # blocks until an item is available
        return return_parameters


class RimtExecutor():
    def __init__(self, send_queue, return_queue):
        self.send_queue = send_queue
        self.return_queue = return_queue

    def execute(self):
        for i in [0]:
            try:
                callback = self.send_queue.get(False)  # doesn't block
                #print('executing')
            except:  # queue.Empty raised when queue is empty (python3.7)
                break
            try:
                #self.return_queue.put(None)
                return_parameters = callback()
                QtCore.QCoreApplication.processEvents()
                self.return_queue.put(return_parameters)
            except Exception as e:
                return_parameters = None
                traceback.print_exc()
                print(e)
        QtCore.QTimer.singleShot(10, self.execute)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    QtCore.QTimer.singleShot(30, main_window.rimt_executor.execute) #<- must be run after the event loop has started (.show()?)
    sys.exit(app.exec_())
