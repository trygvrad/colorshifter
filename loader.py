#!/usr/bin/python
# -*- coding: utf-8 -*-
import PySide2
import pyqtgraph
#import matplotlib._contour
import contourpy
import os
import sys
import numpy
import scipy.interpolate
import scipy.ndimage
import threading
import pathlib
import time
import colorspacious

#try:
# determine if application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(__file__)
i = 0
path = 'colorshifter.py'
if not os.path.exists(path):
    path = str(application_path) + '/colorshifter.py'
    while not os.path.exists(path):
        application_path = application_path + '/..'
        path = str(application_path) + '/colorshifter.py'
        i+=1
        if i>10:
            break
with open(path) as f:
    code = compile(f.read(), path, 'exec')
    exec(code, globals(), locals())
#except Exception as e:
#    print(e)
#    time.sleep(20)
