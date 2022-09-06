#!/usr/bin/env python3
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import  QColorDialog #QTreeWidgetItem
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

        uic.loadUi(path, self)

        self.setObjectName("MainWindow")
        # set icon
        self.setWindowIcon(QtGui.QIcon('iconOri.ico'))


        for img in  [self.image_in, self.original_colorspace, self.shifted_colorspace, self.image_out]:
            img.ui.roiBtn.hide()
            img.ui.menuBtn.hide()
            img.has_img = False
            img.ui.histogram.hide()

        #self.img_0.getHistogramWidget().sigLevelsChanged.connect(self.update_composite_slot)
        self.updating_colors = False

        def dragEnterEvent(ev):
            ev.accept()

        self.image_in.setAcceptDrops(True)
        self.image_in.dropEvent = self.image_in_drop
        self.image_in.dragEnterEvent = dragEnterEvent

        send_queue, return_queue = queue.Queue(), queue.Queue()
        self.rimt = rimt(send_queue, return_queue).rimt
        self.rimt_executor = RimtExecutor(send_queue, return_queue)
        self.files = []
        self.locked = False

        if len(sys.argv)>1:
            file = sys.argv[1]
            self.new_file(file)

        self.save_button.clicked.connect(self.save_clicked)

        self.label.setText(' By Marie Curie fellow Trygve M. R'+chr(int('00E6', 16))+'der. Use at own risk. MIT lisence. https://github.com/trygvrad/colorshifter')

        ### saving
        self.date_today=str(datetime.date.today())
        self.output_int = 1
        while (os.path.exists(f'output/{self.date_today}/{self.output_int}.png')):
            self.output_int += 1
        self.path.setText(f'output/{self.date_today}/{self.output_int}')

    def save_clicked(self,event):
        file = self.path.text()
        os.makedirs('/'.join(file.split('/')[:-1]), exist_ok = True)
        if hasattr(self, 'RGB'):
            import matplotlib.pyplot
            matplotlib.pyplot.imsave(file+'.png', self.RGB.astype(np.uint8))
            matplotlib.pyplot.imsave(file+'.jpg', self.RGB.astype(np.uint8))
            self.mpl_fig.savefig(file+'_stamp.png', transparent=True)


    def image_in_drop(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        file = files[0]
        self.new_file(file)


    def new_file(self,file):
        formats = ['png','jpg','jpeg']
        if file.split('.')[-1] in formats :
            image = np.array(Image.open(file))
            self.image = image
            self.image_in.getImageItem().setImage(np.transpose(image, axes = (1,0,2)))
            self.image_out.getImageItem().setImage(np.transpose(image, axes = (1,0,2)))

            v0 = self.image_in.getView()
            self.image_out.getView().setXLink(v0)
            self.image_out.getView().setYLink(v0)
            r = 45
            self.stamp = colorstamps.stamps.get_const_J(J=70, a=(-1, 1), b=(-1, 1), r=r, l=256, mask='no_mask', rot=0)

            self.original_colorspace.getImageItem().setImage(np.transpose(self.stamp, axes = (1,0,2)))
            self.shifted_colorspace.getImageItem().setImage(np.transpose(self.stamp, axes = (1,0,2)))

            v0 = self.original_colorspace.getView()
            self.shifted_colorspace.getView().setXLink(v0)
            self.shifted_colorspace.getView().setYLink(v0)

            Jab = colorspacious.cspace_convert(image[:,:,:3], "sRGB255", 'CAM02-LCD')

            a_1 = np.linspace(-r, r,256)
            d2 = 0.5*(a_1[1]-a_1[0])
            range = [[a_1[0]-d2, a_1[-1]-d2], [a_1[0]-d2, a_1[-1]-d2]]
            hist, xedges, yedges = np.histogram2d(Jab[:,:,1].ravel(), Jab[:,:,2].ravel(),
                    bins=256, range=range, normed=None, weights=None, density=None)
            hist = scipy.ndimage.gaussian_filter(hist, 3)
            #hist = np.log(hist+0.1)
            hist = np.sqrt(hist)
            self.shifted_colorspace.getImageItem().setImage(np.transpose(hist, axes = (1,0)))
            curves = []

            levels = np.linspace(hist.min(), hist.max(), 7)
            for i, v in enumerate(levels[1:-1]):
                ## generate isocurve with automatic color selection
                c = pyqtgraph.IsocurveItem(data =hist.T,  level=v, pen="w")
                c.setParentItem(self.original_colorspace.getImageItem())  ## make sure isocurve is always correctly displayed over image
                c.setZValue(100)
                curves.append(c)
                self.original_colorspace.addItem(c)



    '''
    def update_composite(self):
        self.updating_colors = False
        if (self.img_0.has_img and self.img_1.has_img):
            imgs = []
            for i, img in enumerate([self.img_0, self.img_1]):
                levels = img.getHistogramWidget().getLevels()
                imgs.append(np.array(255*(img.img-levels[0])/(levels[1]-levels[0]), dtype = int))
                imgs[-1][imgs[-1]>255] = 255
                imgs[-1][ imgs[-1]<0 ] = 0
            if self.num_colors == 2:
                self.RGB = self.stamp[imgs[0], imgs[1]]
                self.composite.getImageItem().setImage(self.RGB, axisOrder = 'row-major')
            else:
                if self.img_2.has_img:
                    img = self.img_2
                    levels = img.getHistogramWidget().getLevels()
                    imgs.append(np.array(255*(img.img-levels[0])/(levels[1]-levels[0]), dtype = int))
                    imgs[-1][imgs[-1]>255] = 255
                    imgs[-1][ imgs[-1]<0 ] = 0
                    self.RGB = self.stamp_not_inv[0][imgs[0]] + self.stamp_not_inv[1][imgs[1]]  + self.stamp_not_inv[2][imgs[2]]
                    if self.invert_check.isChecked():
                        self.RGB = 255-self.RGB
                    self.RGB[self.RGB>255] = 255
                    self.RGB[self.RGB<0 ] = 0

                    self.composite.getImageItem().setImage(self.RGB, axisOrder = 'row-major')
    '''

    @QtCore.pyqtSlot(object)
    def update_composite_slot(self, *args):
        if self.updating_colors == False:
            self.updating_colors = True
            QtCore.QTimer.singleShot(250, self.update_composite)
    '''
    def cmap_changed_slot(self, i):
        cmap = self.cmap_selector.currentText()
        self.mpl_fig.clf()
        self.stamp = self.get_stamp()
        if self.num_colors == 2:
            self.mpl_ax = self.mpl_fig.subplots()
            self.mpl_ax.imshow(self.stamp, origin = 'lower')
            self.mpl_ax.set_xticks([])
            self.mpl_ax.set_yticks([])
            self.img_0.setColorMap(pyqtgraph.ColorMap(pos=np.linspace(0.0, 1.0, 256), color=self.stamp[:,0]))
            self.img_1.setColorMap(pyqtgraph.ColorMap(pos=np.linspace(0.0, 1.0, 256), color=self.stamp[0,:]))
            #self.mpl_fig.tight_layout()
        else: # self.num_colors == 3:
            self.mpl_ax = self.mpl_fig.subplots(1,3)
            for ax in self.mpl_ax:
                ax.set_xticks([])
                ax.set_yticks([])
            self.mpl_ax[0].imshow(self.stamp[0][:,np.newaxis,:]*np.ones((256,20,3), dtype = int), origin = 'lower')
            self.img_0.setColorMap(pyqtgraph.ColorMap(pos=np.linspace(0.0, 1.0, 256), color=self.stamp[0]))
            self.mpl_ax[1].imshow(self.stamp[1][:,np.newaxis,:]*np.ones((256,20,3), dtype = int), origin = 'lower')
            self.img_1.setColorMap(pyqtgraph.ColorMap(pos=np.linspace(0.0, 1.0, 256), color=self.stamp[1]))
            self.mpl_ax[2].imshow(self.stamp[2][:,np.newaxis,:]*np.ones((256,20,3), dtype = int), origin = 'lower')
            self.img_2.setColorMap(pyqtgraph.ColorMap(pos=np.linspace(0.0, 1.0, 256), color=self.stamp[2]))
        # hide ticks
        self.img_0.ui.histogram.gradient.showTicks(False)
        self.img_1.ui.histogram.gradient.showTicks(False)
        self.img_2.ui.histogram.gradient.showTicks(False)
        self.mpl_fig.canvas.draw()
        self.update_composite()

        cmap = self.cmap_selector.currentText()
        if not (cmap == 'Custom' or cmap == '2 Color custom'):
            if self.num_colors == 2:
                self.color_0 = self.stamp[255, 0]
                self.color_1 = self.stamp[0, 255]
    '''
import queue
import functools
class rimt():
    def __init__(self, send_queue, return_queue):
        self.send_queue = send_queue
        self.return_queue = return_queue
        self.main_thread = threading.currentThread()

    def rimt(self, function, *args, **kwargs):
        if threading.currentThread() == self.main_thread:
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
