# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 19:24:05 2018

@author: merzbach

"""

from copy import deepcopy
from datetime import datetime
from functools import wraps
try:
    from IPython import get_ipython
except:
    pass
import imageio
import numpy as np
import os
import sys
import traceback
import time
import types
from warnings import warn

# avoid problems on QT initialization
os.environ['QT_STYLE_OVERRIDE'] = ''

from PyQt5 import QtGui
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QApplication, QCheckBox, QComboBox, QFormLayout, QFrame, QGridLayout, QHBoxLayout, QLabel, \
    QLineEdit, QMainWindow, QPushButton, QShortcut, QSizePolicy, QSpacerItem, QSplitter, QVBoxLayout, QWidget, QFileDialog
from PyQt5.Qt import QImage

import matplotlib
try:
    matplotlib.use('Qt5Agg')
except:
    pass
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.transforms import Bbox

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import colour
except ModuleNotFoundError:
    colour = None
try:
    from torch import Tensor
except ModuleNotFoundError:
    Tensor = type(None)

from pysmtb.utils import crop_bounds, pad

'''
def MyPyQtSlot(*args):
    if len(args) == 0 or isinstance(args[0], types.FunctionType):
        args = []
    @QtCore.pyqtSlot(*args)
    def slotdecorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                func(*args)
            except:
                print("Uncaught Exception in slot")
                traceback.print_exc()
        return wrapper
    return slotdecorator
'''


def iv(*args, **kwargs):
    return IV(*args, **kwargs)


class IV(QMainWindow):
    @staticmethod
    def print_usage():
        print(' ')
        print('hotkeys: ')
        print('a: trigger autoscale')
        print('A: toggle autoscale of [min, max] or ')
        print('   [prctile_low, prctile_high] -> [0, 1], ')
        print('   prctiles can be changed via ctrl+shift+wheel')
        print('c: toggle autoscale on image change')
        print('G: reset gamma to 1')
        print('L: create collage by arranging all images in a ')
        print('   rectangular manner')
        print('O: reset offset to 0')
        print('p: toggle per image auto scale limit computations ')
        print('   (vs. globally over all images)')
        print('S: reset scale to 1')
        print('Z: reset zoom to 100%')
        print('left / right:         switch to next / previous image')
        print('page down / up:       go through images in ~10% steps')
        print('')
        print('wheel:                zoom in / out (inside image axes)')
        print('wheel:                switch to next / previous image')
        print('                      (outside image axes)')
        print('ctrl + wheel:         scale up / down')
        print('shift + wheel:        gamma up / down')
        print('ctrl + shift + wheel: increase / decrease autoscale')
        print('                      percentiles')
        print('left mouse dragged:   pan image')
        print('')

    def __init__(self, *args, **kwargs):
        self.app = QtCore.QCoreApplication.instance()
        if self.app is None:
            self.app = QApplication([''])
        QMainWindow.__init__(self, parent=None)

        self.timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        self.setWindowTitle('iv ' + self.timestamp)

        try:
            shell = get_ipython()
            if shell is not None:
                shell.magic('%matplotlib qt')
        except NameError:
            pass

        def handle_input(inp, images, labels, label=None):
            if isinstance(inp, Tensor):
                # handle torch.Tensor input
                if inp.ndim <= 3:
                    images.append(np.atleast_3d(inp.detach().cpu().numpy()))
                elif inp.ndim == 4:
                    # probably a torch tensor with dimensions [batch, channels, y, x]
                    tmp = inp.detach().cpu().numpy().transpose((2, 3, 1, 0))
                    for imind in range(tmp.shape[3]):
                        images.append(tmp[:, :, :, imind])
                    del tmp
                else:
                    raise Exception('torch tensors can at most have 4 dimensions')

            elif isinstance(inp, np.ndarray):
                if inp.ndim <= 3:
                    images.append(np.atleast_3d(inp))
                elif inp.ndim == 4:
                    # handle 4D numpy.ndarray input by slicing in 4th dimension
                    for imind in range(inp.shape[3]):
                        images.append(inp[:, :, :, imind])
                else:
                    raise Exception('input arrays can be at most 4D')

            else:
                raise Exception('unexpected input type ' + str(type(inp)))

            if label is not None:
                labels.append(label)

        # store list of input images
        self.images = []
        self.labels = kwargs.get('labels', [])
        for arg in args:
            if isinstance(arg, list) or isinstance(arg, tuple):
                for img in arg:
                    handle_input(img, self.images, self.labels)
            elif isinstance(arg, dict):
                for img_label, img in arg.items():
                    handle_input(img, self.images, self.labels, label=img_label)
            else:
                handle_input(arg, self.images, self.labels)

        self.imind = 0  # currently selected image
        self.nims = len(self.images)
        self.scale = kwargs.get('scale', 1.)
        self.gamma = kwargs.get('gamma', 1.)
        self.offset = kwargs.get('offset', 0.)
        self.autoscalePrctiles = kwargs.get('autoscalePrctile', 0.1)
        if np.isscalar(self.autoscalePrctiles):
            self.autoscalePrctiles = np.array([self.autoscalePrctiles, 100. - self.autoscalePrctiles])
        assert len(self.autoscalePrctiles) == 2, 'autoscalePrctiles must have 2 elements!'
        self.autoscalePrctiles[0] = np.clip(self.autoscalePrctiles[0], 0., 50.)
        self.autoscalePrctiles[1] = np.clip(self.autoscalePrctiles[1], 50., 100.)
        self.autoscaleUsePrctiles = kwargs.get('autoscaleUsePrctiles', True)
        self.autoscaleEnabled = kwargs.get('autoscale', True)
        self.autoscaleLower = self.autoscaleEnabled
        self.autoscaleUpper = self.autoscaleEnabled
        self.autoscaleGlobal = kwargs.get('autoscaleGlobal', False)
        self.collageActive = kwargs.get('collage', False)
        self.collageTranspose = kwargs.get('collageTranspose', False)
        self.collageTransposeIms = kwargs.get('collageTransposeIms', False)
        if 'nr' in kwargs and 'nc' in kwargs:
            nr = int(kwargs['nr'])
            nc = int(kwargs['nc'])
        elif 'nr' in kwargs:
            nr = int(kwargs['nr'])
            nc = int(np.ceil(self.nims / nr))
        elif 'nc' in kwargs:
            nc = int(kwargs['nc'])
            nr = int(np.ceil(self.nims / nc))
        else:
            nc = int(np.ceil(np.sqrt(self.nims)))
            nr = int(np.ceil(self.nims / nc))
        self.collage_nr = nr
        self.collage_nc = nc
        self.collage_border_width = kwargs.get('collageBorderWidth', 0)
        self.collage_border_value = kwargs.get('collageBorderValue', 0.)
        self.crop = kwargs.get('crop', False)
        self.crop_global = kwargs.get('crop_global', True)
        self.crop_background = kwargs.get('crop_background', 0)
        self.zoom_factor = 1.1
        self.x_zoom = True
        self.y_zoom = True
        self.x_stop_at_orig = True
        self.y_stop_at_orig = True
        self.annotate = kwargs.get('annotate', False)
        self.annotate_numbers = kwargs.get('annotate_numbers', True)
        self.font_size = kwargs.get('font_size', 12)
        self.font_color = kwargs.get('font_color', 1)
        self.labels = kwargs.get('labels', None)
        if self.labels is not None:
            assert len(self.labels) == len(self.images), 'number of labels %d must match number of images %d'\
                                                         % (len(self.labels), len(self.images))

        # spectral to RGB conversion stuff
        # TODO: expose these
        self.spec_wl0 = 380
        self.spec_wl1 = 730
        self.spec_cmf_names = list(colour.MSDS_CMFS.keys())
        self.spec_illuminant_names = list(colour.SDS_ILLUMINANTS.keys())
        self.spec_cmf_selected_name = 'CIE 1931 2 Degree Standard Observer'
        self.spec_illuminant_selected_name = 'E'

        # image display stuff
        self.ih = None
        self.xmins = []
        self.xmaxs = []
        self.ymins = []
        self.ymaxs = []
        self._compute_crop_bounds()
        self._init_ui()

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self._display_image()
        if self.autoscaleEnabled:
            self.autoscale()
        else:
            self.uiLabelAutoscaleLower.setText('%f' % 0.)
            self.uiLabelAutoscaleUpper.setText('%f' % 1.)
        self.cur_xlims = self.ih.axes.axis()[0:2]
        self.cur_ylims = self.ih.axes.axis()[2:]
        
        self.mouse_down = 0
        self.x_start = 0
        self.y_start = 0
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self._onclick)
        self.cid = self.fig.canvas.mpl_connect('button_release_event', self._onrelease)
        self.cid = self.fig.canvas.mpl_connect('motion_notify_event', self._onmotion)
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.keyPressEvent)
        self.cid = self.fig.canvas.mpl_connect('key_release_event', self.keyReleaseEvent)
        self.cid = self.fig.canvas.mpl_connect('scroll_event', self._onscroll)
        self.alt = False
        self.control = False
        self.shift = False
        self.prev_delta_x = 0
        self.prev_delta_y = 0

        self.ofname = ''  # previous saved image path
        
        self.show()

    def _compute_crop_bounds(self):
        # pre-compute cropping bounds (tight bounding box around non-zero pixels)
        res = crop_bounds(self.images, apply=False, crop_global=self.crop_global, background=self.crop_background)
        self.xmins = res['xmins']
        self.xmaxs = res['xmaxs']
        self.ymins = res['ymins']
        self.ymaxs = res['ymaxs']

    def _init_ui(self):
        self.widget = QWidget()

        self.fig = Figure(dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.widget)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_position(Bbox([[0, 0], [1, 1]]))
        self.ax.set_aspect(1, 'datalim')
        self.ax.set_anchor('NW')
        self.ax.set_clip_on(False)
        self.ax.set_axis_off()
        self._invert_y()

        width = 200

        def _add_widget(w, widget, label, signal=None, callback=None, value=None):
            widget = widget(None if widget == QComboBox else str(value) if label is None else str(label))
            widget.setMaximumWidth(w)
            if label is not None and value is not None:
                if isinstance(widget, QCheckBox):
                    widget.setTristate(False)
                    widget.setCheckState(2 if value else 0)
            if isinstance(widget, QComboBox):
                widget.addItems(value)
            if callback is not None:
                getattr(widget, signal).connect(lambda *args: callback(widget, *args))
            return widget

        self.uiLabelModifiers = QLabel('')
        self.uiLabelModifiers.setMaximumWidth(width)
        self.uiLEScale = _add_widget(width, QLineEdit, None, 'editingFinished', self._callback_line_edit, self.scale)
        self.uiLEGamma = _add_widget(width, QLineEdit, None, 'editingFinished', self._callback_line_edit, self.gamma)
        self.uiLEOffset = _add_widget(width, QLineEdit, None, 'editingFinished', self._callback_line_edit, self.offset)
        self.uiCBAutoscaleLower = _add_widget(width // 2, QCheckBox, 'lower', 'stateChanged', self._callback_check_box, self.autoscaleLower)
        self.uiCBAutoscaleUpper = _add_widget(width // 2, QCheckBox, 'upper', 'stateChanged', self._callback_check_box, self.autoscaleUpper)
        self.uiLEAutoscalePrctileLower = _add_widget(width // 2, QLineEdit, None, 'editingFinished', self._callback_line_edit, self.autoscalePrctiles[0])
        self.uiLEAutoscalePrctileUpper = _add_widget(width // 2, QLineEdit, None, 'editingFinished', self._callback_line_edit, self.autoscalePrctiles[1])
        self.uiLabelAutoscaleLower = _add_widget(width // 2, QLabel, '%f' % 0.)
        self.uiLabelAutoscaleUpper = _add_widget(width // 2, QLabel, '%f' % 1.)
        self.uiCBAutoscaleUsePrctiles = _add_widget(width // 2, QCheckBox, 'prcntiles', 'stateChanged', self._callback_check_box, self.autoscaleUsePrctiles)
        self.uiCBAutoscaleGlobal = _add_widget(width // 2, QCheckBox, 'global', 'stateChanged', self._callback_check_box, self.autoscaleGlobal)
        if self.nims > 1:
            self.uiCBCollageActive = _add_widget(width // 2, QCheckBox, 'enable', 'stateChanged', self._callback_check_box, self.collageActive)
            self.uiCBCollageTranspose = _add_widget(width // 2, QCheckBox, 'transp.', 'stateChanged', self._callback_check_box, self.collageTranspose)
            self.uiCBCollageTransposeIms = _add_widget(width // 2, QCheckBox, 'transp. ims.', 'stateChanged', self._callback_check_box, self.collageTransposeIms)
            self.uiLECollageNr = _add_widget(width // 2, QLineEdit, None, 'editingFinished', self._callback_line_edit, self.collage_nr)
            self.uiLECollageNc = _add_widget(width // 2, QLineEdit, None, 'editingFinished', self._callback_line_edit, self.collage_nc)
            self.uiLECollageBW = _add_widget(width // 2, QLineEdit, None, 'editingFinished', self._callback_line_edit, self.collage_border_width)
            self.uiLECollageBV = _add_widget(width // 2, QLineEdit, None, 'editingFinished', self._callback_line_edit, self.collage_border_value)
        self.uiCBCrop = _add_widget(width // 2, QCheckBox, 'enable', 'stateChanged', self._callback_check_box, self.crop)
        self.uiCBCropGlobal = _add_widget(width // 2, QCheckBox, 'enable', 'stateChanged', self._callback_check_box, self.crop_global)
        self.uiLECropBackground = _add_widget(width // 2, QLineEdit, None, 'editingFinished', self._callback_line_edit, self.crop_background)
        self.uiCBAnnotate = _add_widget(width // 2, QCheckBox, 'enable', 'stateChanged', self._callback_check_box, self.annotate)
        self.uiCBAnnotateNumbers = _add_widget(width // 2, QCheckBox, 'numbers', 'stateChanged', self._callback_check_box, self.annotate_numbers)
        self.uiLEFontSize = _add_widget(width, QLineEdit, None, 'editingFinished', self._callback_line_edit, self.font_size)
        self.uiLEFontColor = _add_widget(width, QLineEdit, None, 'editingFinished', self._callback_line_edit, self.font_color)

        # add spectral to RGB conversion options
        self.uiCBSpecCMFs = _add_widget(width, QComboBox, None, 'activated', self._callback_combobox, self.spec_cmf_names)
        self.uiCBSpecCMFs.setCurrentIndex(np.where([name == self.spec_cmf_selected_name for name in self.spec_cmf_names])[0][0])
        self.uiCBSpecIlluminants = _add_widget(width, QComboBox, None, 'activated', self._callback_combobox, self.spec_illuminant_names)
        self.uiCBSpecIlluminants.setCurrentIndex(np.where([name == self.spec_illuminant_selected_name for name in self.spec_illuminant_names])[0][0])
        self.uiLESpecWL0 = _add_widget(width // 2, QLineEdit, None, 'editingFinished', self._callback_line_edit, self.spec_wl0)
        self.uiLESpecWL1 = _add_widget(width // 2, QLineEdit, None, 'editingFinished', self._callback_line_edit, self.spec_wl1)

        self.uiLabelInfo = QLabel('')
        self.uiLabelInfo.setMaximumWidth(width)
        self._update_info()

        # layout
        form = QGridLayout()
        row = [0]

        def _multicolumn(*widgets):
            hbox = QHBoxLayout()
            for widget in widgets:
                hbox.addWidget(widget)
            return hbox

        def _hdiv():
            frame = QFrame()
            frame.setFixedHeight(3)
            frame.setFrameShape(QFrame.HLine)
            frame.setFrameShadow(QFrame.Sunken)
            return frame

        def _add_row(label, widget=None):
            if widget is None:
                form.addWidget(label, row[0], 0, 1, 2)
            else:
                form.addWidget(label, row[0], 0, 1, 1)
                if isinstance(widget, QWidget):
                    form.addWidget(widget, row[0], 1, 1, 1)
                else:
                    form.addLayout(widget, row[0], 1, 1, 1)
            row[0] += 1

        _add_row(QLabel('modifiers:'), self.uiLabelModifiers)
        _add_row(QLabel('scale:'), self.uiLEScale)
        _add_row(QLabel('gamma:'), self.uiLEGamma)
        _add_row(QLabel('offset:'), self.uiLEOffset)
        _add_row(_hdiv())
        _add_row(QLabel('autoScale:'), _multicolumn(self.uiCBAutoscaleLower, self.uiCBAutoscaleUpper))
        _add_row(QLabel(''), _multicolumn(self.uiCBAutoscaleUsePrctiles, self.uiCBAutoscaleGlobal))
        _add_row(QLabel('percentile:'), _multicolumn(self.uiLEAutoscalePrctileLower, self.uiLEAutoscalePrctileUpper))
        _add_row(QLabel('bounds:'), _multicolumn(self.uiLabelAutoscaleLower, self.uiLabelAutoscaleUpper))
        _add_row(_hdiv())
        if self.nims > 1:
            _add_row(QLabel('collage:'), _multicolumn(self.uiCBCollageActive, self.uiCBCollageTranspose))
            _add_row(QLabel('per img:'), self.uiCBCollageTransposeIms)
            _add_row(QLabel('NR x NC:'), _multicolumn(self.uiLECollageNr, self.uiLECollageNc))
            _add_row(QLabel('BW x BV:'), _multicolumn(self.uiLECollageBW, self.uiLECollageBV))
            _add_row(_hdiv())
        _add_row(QLabel('crop:'), _multicolumn(self.uiCBCrop, self.uiCBCropGlobal))
        _add_row(QLabel('crop bgrnd:'), self.uiLECropBackground)
        _add_row(_hdiv())
        _add_row(QLabel('annotate:'), _multicolumn(self.uiCBAnnotate, self.uiCBAnnotateNumbers))
        _add_row(QLabel('font size:'), self.uiLEFontSize)
        _add_row(QLabel('font value:'), self.uiLEFontColor)
        _add_row(_hdiv())
        _add_row(QLabel('info:'), self.uiLabelInfo)
        _add_row(_hdiv())

        any_spectral = np.any([im.ndim > 2 and im.shape[2] > 3 for im in self.images])
        if any_spectral:
            _add_row(QLabel('Illum.'), self.uiCBSpecIlluminants)
            _add_row(QLabel('CMF'), self.uiCBSpecCMFs)
            _add_row(QLabel('WLs'), _multicolumn(self.uiLESpecWL0, self.uiLESpecWL1))
            _add_row(_hdiv())

        # fixed "footer" at bottom right with copy & save buttons
        width_bottom = width + 100
        self.uiPBPrevImg = _add_widget(width_bottom // 2, QPushButton, '&prev', 'clicked', self._callback_push_button)
        self.uiPBNextImg = _add_widget(width_bottom // 2, QPushButton, '&next', 'clicked', self._callback_push_button)
        self.uiPBCopyClipboard = _add_widget(width_bottom // 2, QPushButton, '&copy', 'clicked', self._callback_push_button)
        self.uiPBCopyClipboardZoomed = _add_widget(width_bottom // 2, QPushButton, 'copy &zoomed', 'clicked', self._callback_push_button)
        self.uiPBSave = _add_widget(width_bottom // 2, QPushButton, '&save', 'clicked', self._callback_push_button)
        self.uiPBSaveZoomed = _add_widget(width_bottom // 2, QPushButton, 'sa&ve zoomed', 'clicked', self._callback_push_button)
        self.uiPBSaveCanvas = _add_widget(width_bottom // 2, QPushButton, 's&ave canvas', 'clicked', self._callback_push_button)

        row_bottom = 0
        form_bottom = QGridLayout()
        form_bottom.addWidget(self.uiPBPrevImg, row_bottom, 0)
        form_bottom.addWidget(self.uiPBNextImg, row_bottom, 1)
        row_bottom += 1
        form_bottom.addWidget(_hdiv(), row_bottom, 0, 1, 2)
        row_bottom += 1
        form_bottom.addWidget(self.uiPBCopyClipboard, row_bottom, 0)
        form_bottom.addWidget(self.uiPBCopyClipboardZoomed, row_bottom, 1)
        row_bottom += 1
        form_bottom.addWidget(_hdiv(), row_bottom, 0, 1, 2)
        row_bottom += 1
        form_bottom.addWidget(self.uiPBSave, row_bottom, 0)
        form_bottom.addWidget(self.uiPBSaveZoomed, row_bottom, 1)
        row_bottom += 1
        form_bottom.addWidget(self.uiPBSaveCanvas, row_bottom, 0)
        row_bottom += 1

        vbox = QVBoxLayout()
        vbox.addLayout(form)
        vbox.addItem(QSpacerItem(1, 1, vPolicy=QSizePolicy.Expanding))
        vbox.addLayout(form_bottom)
        
        hbox_canvas = QHBoxLayout()
        hbox_canvas.addWidget(self.canvas)
        hbox_canvas.addLayout(vbox)
        
        self.widget.setLayout(hbox_canvas)
        self.setCentralWidget(self.widget)
        
        # make image canvas expand with window
        sp = self.canvas.sizePolicy()
        sp.setHorizontalStretch(1)
        sp.setVerticalStretch(1)
        self.canvas.setSizePolicy(sp)
        
        self.ih = self.ax.imshow(np.zeros(self.get_img().shape[:2] + (3,)), origin='upper')
        self.ax.set_position(Bbox([[0, 0], [1, 1]]))
        self._invert_y()

        # keyboard shortcuts
        # scaleShortcut = QShortcut(QKeySequence('Ctrl+Shift+a'), self.widget)
        # scaleShortcut.activated.connect(self.autoscale)
        close_shortcut = QShortcut(QKeySequence('Escape'), self.widget)
        close_shortcut.activated.connect(self.close)
        QShortcut(QKeySequence('a'), self.widget).activated.connect(self.autoscale)
        QShortcut(QKeySequence('Shift+a'), self.widget).activated.connect(self._toggle_autoscale_use_prctiles)

    def _callback_line_edit(self, ui, *args):
        try:
            tmp = float(ui.text())
        except:
            return
        
        if ui == self.uiLEScale:
            self.set_scale(tmp)
        elif ui == self.uiLEGamma:
            self.set_gamma(tmp)
        elif ui == self.uiLEOffset:
            self.set_offset(tmp)
        elif ui == self.uiLEAutoscalePrctileLower:
            self.autoscalePrctiles[0] = np.clip(tmp, 0., 50.)
            self.autoscale()
        elif ui == self.uiLEAutoscalePrctileUpper:
            self.autoscalePrctiles[1] = np.clip(tmp, 50., 100.)
            self.autoscale()
        elif hasattr(self, 'uiLECollageNr') and ui == self.uiLECollageNr:
            self.collage_nr = int(tmp)
            self._display_collage()
        elif hasattr(self, 'uiLECollageNc') and ui == self.uiLECollageNc:
            self.collage_nc = int(tmp)
            self._display_collage()
        elif hasattr(self, 'uiLECollageBW') and ui == self.uiLECollageBW:
            self.collage_border_width = int(tmp)
            self._display_collage()
        elif hasattr(self, 'uiLECollageBV') and ui == self.uiLECollageBV:
            self.collage_border_value = float(tmp)
            self._display_collage()
        elif ui == self.uiLEFontSize:
            self.font_size = int(tmp)
            self._display_image()
        elif ui == self.uiLEFontColor:
            self.font_color = tmp
            self._display_image()
        elif ui == self.uiLECropBackground:
            self.crop_background = tmp
            self._compute_crop_bounds()
            self._display_image()
        elif hasattr(self, 'uiLESpecWL0') and ui == self.uiLESpecWL0:
            self.spec_wl0 = tmp
            self._display_image()
        elif hasattr(self, 'uiLESpecWL1') and ui == self.uiLESpecWL1:
            self.spec_wl1 = tmp
            self._display_image()

    def _callback_check_box(self, ui, state):
        print(ui, state)
        if ui == self.uiCBAutoscaleUsePrctiles:
            self.autoscaleUsePrctiles = bool(state)
            if self.autoscaleEnabled:
                self.autoscale()
        elif ui == self.uiCBAutoscaleGlobal:
            self.autoscaleGlobal = bool(state)
            if self.autoscaleEnabled:
                self.autoscale()
        elif ui == self.uiCBAutoscaleLower:
            self.autoscaleLower = bool(state)
            self.autoscaleEnabled = self.autoscaleLower or self.autoscaleUpper
            if self.autoscaleEnabled:
                self.autoscale()
        elif ui == self.uiCBAutoscaleUpper:
            self.autoscaleUpper = bool(state)
            self.autoscaleEnabled = self.autoscaleLower or self.autoscaleUpper
            if self.autoscaleEnabled:
                self.autoscale()
        elif hasattr(self, 'uiCBCollageActive') and ui == self.uiCBCollageActive:
            self.collageActive = bool(state)
            self._display_image()
        elif hasattr(self, 'uiCBCollageTranspose') and ui == self.uiCBCollageTranspose:
            self.collageTranspose = bool(state)
            self._display_image()
        elif hasattr(self, 'uiCBCollageTransposeIms') and ui == self.uiCBCollageTransposeIms:
            self.collageTransposeIms = bool(state)
            self._display_image()
        elif ui == self.uiCBCrop:
            self.crop = bool(state)
            self._display_image()
        elif ui == self.uiCBCropGlobal:
            self.crop_global = bool(state)
            self._compute_crop_bounds()
            self._display_image()
        elif ui == self.uiCBAnnotate:
            self.annotate = bool(state)
            print('annotate set to ' + str(self.annotate), 'type(state): ', type(state), ', state: ', state)
            self._display_image()
        elif ui == self.uiCBAnnotateNumbers:
            self.annotate_numbers = bool(state)
            self._display_image()

    def _callback_combobox(self, ui, index):
        if ui == self.uiCBSpecCMFs:
            self.spec_cmf_selected_name = self.uiCBSpecCMFs.currentText()
            self._display_image()
        elif ui == self.uiCBSpecIlluminants:
            self.spec_illuminant_selected_name = self.uiCBSpecIlluminants.currentText()
            self._display_image()

    def _callback_push_button(self, ui, *args):
        if ui == self.uiPBCopyClipboard:
            self.copy_to_clipboard()
        elif ui == self.uiPBCopyClipboardZoomed:
            self.copy_to_clipboard_zoomed()
        elif ui == self.uiPBSave:
            self.save(zoomed=False)
        elif ui == self.uiPBSaveZoomed:
            self.save(zoomed=True)
        elif ui == self.uiPBSaveCanvas:
            self.save(canvas=True)
        elif ui == self.uiPBPrevImg:
            self.switch_image(-1)
        elif ui == self.uiPBNextImg:
            self.switch_image(1)
    
    def get_img(self, i=None, tonemap=False, decorate=False):
        """return i-th image, optionally tonemapped and decorated"""
        if i is None:
            i = self.imind
        im = self.images[i]
        if self.crop:
            im = im[self.ymins[i]:self.ymaxs[i], self.xmins[i]:self.xmaxs[i], :]
        if tonemap:
            im = self.tonemap(im)
        if decorate and self.annotate:
            im = self.decorate(im=im, i=i)
        return im
    
    def get_imgs(self, tonemap=False, decorate=False):
        """return all images in a list, optionally tonemapped and decorated"""
        return [self.get_img(ind, tonemap=tonemap, decorate=decorate) for ind in range(len(self.images))]

    def decorate(self, im, i=None, label=''):
        """add annotation to an image"""
        if i is None:
            i = self.imind
        if self.annotate:
            from pysmtb.utils import annotate_image
            if self.annotate_numbers:
                label += str(i) + ' '
            if self.labels is not None:
                label += self.labels[i]
            if im.shape[2] == 3:
                im = annotate_image(im, label, font_size=self.font_size, font_color=self.font_color)
            else:
                im = annotate_image(im[:, :, 0], label, font_size=self.font_size, font_color=self.font_color, stroke_color=np.clip(1.-self.font_color, 0, 1))
        return im
    
    def autoscale(self):
        """autoscale between user-selected percentiles"""
        if self.autoscaleUsePrctiles:
            if self.autoscaleGlobal:
                limits = [np.percentile(image, self.autoscalePrctiles)
                          for image in self.get_imgs(tonemap=False, decorate=False)]
                lower = np.min([lims[0] for lims in limits])
                upper = np.max([lims[1] for lims in limits])
            else:
                lower, upper = np.percentile(self.get_img(tonemap=False, decorate=False), self.autoscalePrctiles)
        else:
            if self.autoscaleGlobal:
                ims = self.get_imgs(tonemap=False, decorate=False)
                lower = np.min([np.min(image) for image in ims])
                upper = np.max([np.max(image) for image in ims])
            else:
                im = self.get_img(tonemap=False, decorate=False)
                lower = np.min(im)
                upper = np.max(im)
        if self.autoscaleLower:
            self.set_offset(lower, False)
            self.uiLabelAutoscaleLower.setText('%f' % lower)
        if self.autoscaleUpper:
            self.set_scale(1. / (upper - lower), True)
            self.uiLabelAutoscaleUpper.setText('%f' % upper)

    def _toggle_autoscale_use_prctiles(self):
        self.autoscaleUsePrctiles = not self.autoscaleUsePrctiles
        self.autoscale()

    def _display_collage(self):
        # arrange all images in a collage and display them
        if self.collage_nr * self.collage_nc < self.nims:
            nc = int(np.ceil(np.sqrt(self.nims)))
            nr = int(np.ceil(self.nims / nc))
            self.collage_nr = nr
            self.collage_nc = nc
            self.uiLECollageNr.blockSignals(True)
            self.uiLECollageNc.blockSignals(True)
            self.uiLECollageNr.setText(str(nr))
            self.uiLECollageNc.setText(str(nc))
            self.uiLECollageNr.blockSignals(False)
            self.uiLECollageNc.blockSignals(False)
        else:
            nc = self.collage_nc
            nr = self.collage_nr
        
        # pad array so it matches the product nc * nr
        padding = nc * nr - self.nims
        ims = self.get_imgs(tonemap=True, decorate=True)
        h = np.max([im.shape[0] for im in ims])
        w = np.max([im.shape[1] for im in ims])
        num_channels = np.max([im.shape[2] for im in ims])
        ims = [pad(im, new_width=w, new_height=h, new_num_channels=num_channels) for im in ims]
        ims += [self.collage_border_value * np.ones((h, w, num_channels))] * padding
        coll = np.stack(ims, axis=3)
        coll = np.reshape(coll, (h, w, num_channels, nr, nc))
        # 0  1  2   3   4
        # y, x, ch, ro, co
        if self.collage_border_width:
            # pad each patch by border if requested
            coll = np.append(coll, self.collage_border_value * np.ones((self.collage_border_width, ) + coll.shape[1 : 5]), axis=0)
            coll = np.append(coll, self.collage_border_value * np.ones((coll.shape[0], self.collage_border_width) + coll.shape[2 : 5]), axis=1)
        if self.collageTranspose:
            nim0 = nc
            nim1 = nr
            if self.collageTransposeIms:
                dim0 = w
                dim1 = h
                #                          nr w  nc h  ch
                coll = np.transpose(coll, (4, 1, 3, 0, 2))
            else:
                dim0 = h
                dim1 = w
                #                          nr h  nc w  ch
                coll = np.transpose(coll, (4, 0, 3, 1, 2))
        else:
            nim0 = nr
            nim1 = nc
            if self.collageTransposeIms:
                dim0 = w
                dim1 = h
                #                          nc w  nr h  ch
                coll = np.transpose(coll, (3, 1, 4, 0, 2))
            else:
                dim0 = h
                dim1 = w
                #                          nc h  nr w  ch
                coll = np.transpose(coll, (3, 0, 4, 1, 2))
        coll = np.reshape(coll, ((dim0 + self.collage_border_width) * nim0, (dim1 + self.collage_border_width) * nim1, num_channels))

        self.ax.clear()
        if coll.dtype == np.float16:
            coll = coll.astype(np.float32)
        self.ih = self.ax.imshow(coll, origin='upper')
        
        height, width = self.ih.get_size()
        limits = (-0.5, width - 0.5, -0.5, height - 0.5)
        self.ax.set(xlim=limits[0:2], ylim=limits[2:4])
        self._invert_y()
        self.fig.canvas.draw()

    def _switch_to_single_image(self):
        # reset canvas to show a single image instead of a collage
        if self.collageActive:
            self.ax.clear()
            self.ih = self.ax.imshow(np.zeros(self.get_img(tonemap=True).shape[:3]), origin='upper')
        self.collageActive = False
        
    def reset_zoom(self):
        """reset zoom factor to 1, i.e. show the entire image"""
        height, width = self.ih.get_size()
        limits = (-0.5, width - 0.5, -0.5, height - 0.5)
        self.ih.axes.axis(limits)
        self.ax.set_position(Bbox([[0, 0], [1, 1]]))
        self._invert_y()
        self.fig.canvas.draw()
        
    def zoom(self, pos, factor):
        """zoom on specific position in image by specified zoom factor"""
        limits = self.ih.axes.axis()
        xlim = limits[0:2]
        ylim = limits[2:]
        
        # compute interval lengths left, right, below and above cursor
        left = pos[0] - xlim[0]
        right = xlim[1] - pos[0]
        below = pos[1] - ylim[0]
        above = ylim[1] - pos[1]
        
        # zoom in or out
        if self.x_zoom:
            xlim = [pos[0] - factor * left, pos[0] + factor * right]
        if self.y_zoom:
            ylim = [pos[1] - factor * below, pos[1] + factor * above]
        
        # no zooming out beyond original zoom level
        height, width = self.ih.get_size()
        
        if self.x_stop_at_orig:
            xlim = [np.maximum(-0.5, xlim[0]), np.minimum(width - 0.5, xlim[1])]
        
        if self.y_stop_at_orig:
            ylim = [np.maximum(-0.5, ylim[0]), np.minimum(height - 0.5, ylim[1])]
        
        # update axes
        if xlim[0] != xlim[1] and ylim[0] != ylim[1]:
            limits = (xlim[0], xlim[1], ylim[0], ylim[1])
            self.ih.axes.axis(limits)
            self._invert_y()
            self.ax.set_position(Bbox([[0, 0], [1, 1]]))
            self.fig.canvas.draw()
        return

    def overlay_pixel_values(self):
        # display overlay at cursor position showing numeric pixel values
        kids = self.ax.get_children()
        for kid in kids:
            if isinstance(kid, matplotlib.text.Text):
                kid.set_visible(False)
        lims = self.ih.axes.axis()
        inds = (np.array(lims) + 0.5).astype(np.int32)
        im = self.ih.get_array()
        im = im[inds[3]: inds[2], inds[0]: inds[1], :]
        xs = np.r_[inds[0]: inds[1]]
        ys = np.r_[inds[3]: inds[2]]

        for xi, x0 in enumerate(xs):
            for yi, y0 in enumerate(ys):
                pixel = im[yi, xi, :]
                color = (pixel + 0.5) % 1.
                if im.ndim == 3 and im.shape[2] == 3:
                    th = self.ax.text(x0, y0 + 1.0, '% 6.3f\n% 6.3f\n% 6.3f\n' % tuple(pixel), fontsize=10, color=color)
                else:
                    th = self.ax.text(x0, y0 + 1.0, '% 6.3f' % im[0, 0], fontsize=10, color=color)
        self.fig.canvas.draw()

    def tonemap(self, im):
        """apply simple scaling & gamma based tonemapping to HDR image, convert spectral to RGB"""
        # TODO: add color mapping for single channel images
        if isinstance(im, np.matrix):
            im = np.array(im)
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 2:
            im = np.concatenate((im, np.zeros((im.shape[0], im.shape[1], 1), dtype=im.dtype)), axis=2)
        elif im.shape[2] != 3:
            # project to RGB
            if colour is None:
                raise NotImplemented('please install the colour-science package')

            wl_range = self.spec_wl1 - self.spec_wl0
            spec_shape = colour.SpectralShape(self.spec_wl0, self.spec_wl1, wl_range / (im.shape[2] - 1))

            illuminant = deepcopy(colour.SDS_ILLUMINANTS[self.spec_illuminant_selected_name])
            illuminant = illuminant.align(shape=spec_shape)
            cmfs = deepcopy(colour.MSDS_CMFS[self.spec_cmf_selected_name])
            cmfs = cmfs.align(shape=spec_shape)
            im = colour.msds_to_XYZ(im, cmfs, illuminant, method='Integration', shape=spec_shape)
            if self.spec_cmf_selected_name.lower().startswith('cie'):
                im = colour.XYZ_to_sRGB(im / 100)
            else:
                im /= 100
        return np.clip((im - self.offset) * self.scale, 0, 1) ** (1. / self.gamma)

    def switch_image(self, delta=1, redraw=True):
        """set index to previous or next image, optionally skip redrawing of canvas (and thus the actual image display)"""
        self._switch_to_single_image()
        self.imind = int(np.mod(self.imind + delta, self.nims))
        self._update_info()
        if self.autoscaleEnabled:
            self.autoscale()
        if redraw:
            self._display_image()

    def _display_image(self):
        # display collage or single image, resetting the axes (zoom) when necessary
        if self.collageActive:
            self._display_collage()
            self.setWindowTitle('iv ' + self.timestamp + ' %d x %d collage (#ims: %d)'
                                % (self.collage_nr, self.collage_nc, self.nims))
        else:
            if self.nims > 1:
                self.uiCBCollageActive.blockSignals(True)
                self.uiCBCollageActive.setChecked(False)
                self.uiCBCollageActive.blockSignals(False)
            height, width = self.ih.get_size()
            im = self.get_img(tonemap=True, decorate=True)
            if im.dtype == np.float16:
                # matplotlib rejects float16...
                im = im.astype(np.float32)
            if height != im.shape[0] or width != im.shape[1]:
                # image size changed, create new axes
                self.ax.clear()
                self.ih = self.ax.imshow(im)
            else:
                self.ih.set_data(im)
            # TODO: add "keep axis" checkbox that disables the following on chaning images so that zoom & pan can be sustained
            #height, width = self.ih.get_size()
            #lims = (-0.5, width - 0.5, -0.5, height - 0.5)
            #self.ax.set(xlim = lims[0:2], ylim = lims[2:4])
            self._invert_y()
            self.fig.canvas.draw()
            self.setWindowTitle('iv ' + self.timestamp + ' %d / %d' % (self.imind + 1, self.nims))

    def _update_info(self):
        self.uiLabelInfo.setText(('image: %d / %d\nimage size:\n' % (self.imind + 1, self.nims)) + str(self.get_img().shape))

    def _invert_y(self):
        try:
            self.ax.get_yaxis().set_inverted(True)
        except AttributeError:
            self.ax.invert_yaxis()

    def set_scale(self, scale, redraw=True):
        self.scale = scale
        self.uiLEScale.setText(str(self.scale))
        self.uiLabelAutoscaleLower.setText('%f' % self.offset)
        self.uiLabelAutoscaleUpper.setText('%f' % ((1 / self.scale) + self.offset))
        if redraw:
            self._display_image()

    def set_gamma(self, gamma, redraw=True):
        self.gamma = gamma
        self.uiLEGamma.setText(str(self.gamma))
        if redraw:
            self._display_image()
    
    def set_offset(self, offset, redraw=True):
        self.offset = offset
        self.uiLEOffset.setText(str(self.offset))
        self.uiLabelAutoscaleLower.setText('%f' % self.offset)
        self.uiLabelAutoscaleUpper.setText('%f' % ((1 / self.scale) + self.offset))
        if redraw:
            self._display_image()

    def _onclick(self, event):
        if event.dblclick:
            self.reset_zoom()
            self.mouse_down ^= event.button
        elif event.inaxes:
            self.x_start = event.xdata
            self.y_start = event.ydata
            self.prev_delta_x = 0
            self.prev_delta_y = 0
            self.cur_xlims = self.ih.axes.axis()[0: 2]
            self.cur_ylims = self.ih.axes.axis()[2:]
            self.mouse_down |= event.button
            
    def _onrelease(self, event):
        self.mouse_down ^= event.button
            
    def _onmotion(self, event):
        if self.mouse_down == 1 and event.inaxes:
            delta_x = self.x_start - event.xdata
            delta_y = self.y_start - event.ydata
            self.ih.axes.axis((self.cur_xlims[0] + delta_x,
                               self.cur_xlims[1] + delta_x, 
                               self.cur_ylims[0] + delta_y,
                               self.cur_ylims[1] + delta_y))
            self.fig.canvas.draw()
            self.x_start += (delta_x - self.prev_delta_x)
            self.y_start += (delta_y - self.prev_delta_y)
            self.prev_delta_x = delta_x
            self.prev_delta_y = delta_y

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        mod = event.modifiers()
        if key == Qt.Key_Question:  # ?
            IV.print_usage()
        elif key == Qt.Key_A:  # a
            # trigger autoscale
            self.autoscale()
            return
        elif key == Qt.Key_A and mod == Qt.Key_Shift:  # A
            # toggle autoscale between user-selected percentiles or min-max
            self.autoscaleUsePrctiles = not self.autoscaleUsePrctiles
            self.autoscale()
            return
        elif key == Qt.Key_C:
            # toggle on-change autoscale
            self.autoscaleEnabled = not self.autoscaleEnabled
            print('on-change autoscaling is %s' % ('on' if self.autoscaleEnabled else 'off'))
        elif key == Qt.Key_G:
            self.gamma = 1.
        elif key == Qt.Key_L:
            # update axes for single image dimensions
            if self.collageActive:
                self._switch_to_single_image()
            else:
                # toggle showing collage
                self.collageActive = not self.collageActive
            # also disable per-image scaling limit computation
            self.autoscaleGlobal = not self.autoscaleGlobal
        elif key == Qt.Key_O:
            self.offset = 0.
        elif key == Qt.Key_P:
            self.autoscaleGlobal = not self.autoscaleGlobal
            print('per-image scaling is %s' % ('on' if self.autoscaleGlobal else 'off'))
            self.autoscale()
        elif key == Qt.Key_S:
            self.scale = 1.
        elif key == Qt.Key_Z:
            # reset zoom
            self.ih.axes.autoscale(True)
        elif key == Qt.Key_Alt:
            self.alt = True
            self.uiLabelModifiers.setText('alt: %d, ctrl: %d, shift: %d' % (self.alt, self.control, self.shift))
            return
        elif key == Qt.Key_Control:
            self.control = True
            self.uiLabelModifiers.setText('alt: %d, ctrl: %d, shift: %d' % (self.alt, self.control, self.shift))
            return
        elif key == Qt.Key_Shift:
            self.shift = True
            self.uiLabelModifiers.setText('alt: %d, ctrl: %d, shift: %d' % (self.alt, self.control, self.shift))
            return
        elif key == Qt.Key_Left:
            self.switch_image(-1, False)
        elif key == Qt.Key_Right:
            self.switch_image(1, False)
        else:
            return
        self._display_image()

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key_Alt:
            self.alt = False
        elif key == Qt.Key_Control:
            self.control = False
        elif key == Qt.Key_Shift:
            self.shift = False
        self.uiLabelModifiers.setText('alt: %d, ctrl: %d, shift: %d' % (self.alt, self.control, self.shift))
    
    def _onscroll(self, event):
        if self.control and self.shift:
            # autoscale percentiles
            self.autoscalePrctiles[0] = np.clip(self.autoscalePrctiles[0] / np.power(1.1, event.step), 0., 50.)
            self.autoscalePrctiles[1] = np.clip(self.autoscalePrctiles[1] * np.power(1.1, event.step), 50., 100.)
            print('auto percentiles: [%3.5f, %3.5f]' % (self.autoscalePrctiles[0], self.autoscalePrctiles[1]))
            self.autoscaleUsePrctiles = True
            self.autoscale()
        elif self.control:
            # scale
            self.set_scale(self.scale * np.power(1.1, event.step))
        elif self.shift:
            # gamma
            self.set_gamma(self.gamma * np.power(1.1, event.step))
        else:
            x = event.xdata
            y = event.ydata
            h, w, = self.ih.get_size()
            x0, x1 = -0.5, w - 0.5
            y0, y1 = -0.5, h - 0.5
            if event.inaxes and x0 <= x <= x1 and y0 <= y <= y1:
                # zoom when inside image axes
                factor = np.power(self.zoom_factor, -event.step)
                self.zoom([x, y], factor)
                return
            else:
                # scroll through images when outside of axes
                self.switch_image(-event.step, False)
        self._display_image()

    def copy_to_clipboard(self):
        im = (255 * self.ih.get_array()).astype(np.uint8)
        h, w, nc = im.shape[:3]
        im = QImage(im.tobytes(), w, h, nc * w, QImage.Format_RGB888)
        c = QApplication.clipboard()
        c.setImage(im)

    def copy_to_clipboard_zoomed(self):
        extent = self.ax.get_window_extent()
        width_axes = extent.width
        height_axes = extent.height

        im = QImage(self.canvas.grab())
        im = im.copy(0, 0, int(width_axes), int(height_axes))
        c = QApplication.clipboard()
        c.setImage(im)

    def save(self, ofname=None, zoomed=False, canvas=True):
        try:
            if ofname is None:
                dialog = QFileDialog()
                ofname = dialog.getSaveFileName(parent=self,
                                                caption='file save path',
                                                directory=os.path.split(self.ofname)[0])[0]
            if ofname is None or not len(ofname):
                return
            self.ofname = ofname
            im = self.get_img()
            if zoomed:
                h, w = im.shape[:2]
                limits = self.ax.axis()
                x0 = np.max([0, int(limits[0] + 0.5)])
                x1 = np.min([w, int(limits[1] + 0.5)])
                y0 = np.max([0, int(limits[3] + 0.5)])
                y1 = np.min([h, int(limits[2] + 0.5)])
                image = im[y0:y1, x0:x1, :]
            elif canvas:
                from pysmtb.utils import qimage_to_np
                im = QImage(self.canvas.grab())
                # im = im.convertToFormat(QImage.Format_RGB888)
                image = qimage_to_np(im)

                image = image[:, :, -2::-1]

                # get only image content, not the white stuff from the canvas
                extent = self.ax.get_window_extent()
                width_axes = extent.width
                height_axes = extent.height
                image = image[: int(height_axes), : int(width_axes), :]
            else:
                image = np.array(self.ih.get_array())
            imageio.imwrite(ofname, image)
        except Exception as e:
            warn(str(e))
