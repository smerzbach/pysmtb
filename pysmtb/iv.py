"""
Created on Thu Oct 18 19:24:05 2018

@author: Sebastian Merzbach

setup:
pip install -r requirements.txt
pip install pysmtb

or manually install:
  - colour-science
  - imageio
  - matplotlib
  - numpy
  - OpenEXR
  - PyQt5
  - pysmtb

On a Ubuntu, you might have to first run:
sudo apt-get install libopenexr-dev openexr zlib1g-dev

usage from command line:

    python iv.py -i image.exr
    python iv.py -i image.exr --autoscale 0 --scale 2
    python iv.py -i image1.exr image2.exr --autoscale 0 --scale 2 --collage 1
    python iv.py -i *.exr --autoscale 1 --autoscaleGlobal 1 --collage 1 --nr 5 --nc 7

usage from code:

from pysmtb.iv import iv
v = iv(image1, image2, ...)
v = iv([image1, image2, image3], image4, image5, ...)
v = iv(images)  # image being H x W x C x N np.ndarray or torch.Tensor
v = iv(..., autoscale=True, autoscaleGlobal=True)
v = iv(..., autoscale=True, autoscaleGlobal=True, collage=True)

TODO: iv currently doesn't support specifying wavelength channels per image

"""

from copy import deepcopy
from datetime import datetime
from functools import wraps
try:
    from IPython import get_ipython
except:
    pass
import imageio
import json
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
import matplotlib.cm as cm
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

from pysmtb.utils import crop_bounds, pad, collage, qimage_to_np

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
        self.collage_tight = kwargs.get('collageTight', kwargs.get('collage_tight', True))
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
        self.has_alpha = kwargs.get('has_alpha', False)
        self.blend_alpha = kwargs.get('blend_alpha', False)
        self.background = kwargs.get('background', 0.0)
        self.crop = kwargs.get('crop', False)
        self.crop_global = kwargs.get('crop_global', True)
        self.crop_background = kwargs.get('crop_background', 0.0)
        self.zoom_factor = 1.1
        self.x_zoom = True
        self.y_zoom = True
        self.x_stop_at_orig = True
        self.y_stop_at_orig = True
        self.annotate = kwargs.get('annotate', False)
        self.annotate_numbers = kwargs.get('annotate_numbers', True)
        self.font_size = kwargs.get('font_size', 12)
        self.font_color = kwargs.get('font_color', 1)
        if len(self.labels) == 0:
            self.labels = None
        if self.labels is not None:
            assert len(self.labels) == len(self.images), 'number of labels %d must match number of images %d'\
                                                         % (len(self.labels), len(self.images))

        # stores np.ndarray, QImage and QApplication.clipboard() objects to prevent garbage collection when copying
        # canvas or image to clipboard
        self.clipboard_image = None
        self.clipboard_qimage = None
        self.clipboard = None

        # spectral to RGB conversion stuff
        # TODO: expose these
        self.spec_wl0 = 380
        self.spec_wl1 = 730
        if colour is not None:
            self.spec_cmf_names = list(colour.MSDS_CMFS.keys())
            self.spec_illuminant_names = list(colour.SDS_ILLUMINANTS.keys())
        else:
            self.spec_cmf_names = ['pip install colour-science']
            self.spec_illuminant_names = ['pip install colour-science']
        self.spec_cmf_selected_name = 'CIE 1931 2 Degree Standard Observer'
        self.spec_illuminant_selected_name = 'E'

        # colormapping for scalar-valued inputs
        self.cm_names = list(cm._cmap_registry.keys())
        self.cm_name_selected = kwargs.get('cm', 'gray')

        # image display stuff
        self.ih = None
        self.xmins = []
        self.xmaxs = []
        self.ymins = []
        self.ymaxs = []
        self.overlay_ths = []
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
        self.repaint()
        self.canvas.draw()

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
            self.uiCBCollageTight = _add_widget(width // 2, QCheckBox, 'tight', 'stateChanged', self._callback_check_box, self.collage_tight)
            self.uiCBCollageTranspose = _add_widget(width // 2, QCheckBox, 'transp.', 'stateChanged', self._callback_check_box, self.collageTranspose)
            self.uiCBCollageTransposeIms = _add_widget(width // 2, QCheckBox, 'transp. ims.', 'stateChanged', self._callback_check_box, self.collageTransposeIms)
            self.uiLECollageNr = _add_widget(width // 2, QLineEdit, None, 'editingFinished', self._callback_line_edit, self.collage_nr)
            self.uiLECollageNc = _add_widget(width // 2, QLineEdit, None, 'editingFinished', self._callback_line_edit, self.collage_nc)
            self.uiLECollageBW = _add_widget(width // 2, QLineEdit, None, 'editingFinished', self._callback_line_edit, self.collage_border_width)
            self.uiLECollageBV = _add_widget(width // 2, QLineEdit, None, 'editingFinished', self._callback_line_edit, self.collage_border_value)
        self.uiCBHasAlpha = _add_widget(width // 2, QCheckBox, 'available', 'stateChanged', self._callback_check_box, self.has_alpha)
        self.uiCBBlendAlpha = _add_widget(width // 2, QCheckBox, 'blend', 'stateChanged', self._callback_check_box, self.blend_alpha)
        self.uiCBCrop = _add_widget(width // 2, QCheckBox, 'enable', 'stateChanged', self._callback_check_box, self.crop)
        self.uiCBCropGlobal = _add_widget(width // 2, QCheckBox, 'global', 'stateChanged', self._callback_check_box, self.crop_global)
        self.uiLECropBackground = _add_widget(width // 2, QLineEdit, None, 'editingFinished', self._callback_line_edit, self.crop_background)
        self.uiLEBackground = _add_widget(width // 2, QLineEdit, None, 'editingFinished', self._callback_line_edit, self.background)
        self.uiCBAnnotate = _add_widget(width // 2, QCheckBox, 'enable', 'stateChanged', self._callback_check_box, self.annotate)
        self.uiCBAnnotateNumbers = _add_widget(width // 2, QCheckBox, 'numbers', 'stateChanged', self._callback_check_box, self.annotate_numbers)
        self.uiLEFontSize = _add_widget(width, QLineEdit, None, 'editingFinished', self._callback_line_edit, self.font_size)
        self.uiLEFontColor = _add_widget(width, QLineEdit, None, 'editingFinished', self._callback_line_edit, self.font_color)

        # add colormap options
        self.uiCBColormaps = _add_widget(width / 2, QComboBox, None, 'activated', self._callback_combobox, self.cm_names)
        cm_ind = np.where([name == self.cm_name_selected for name in self.cm_names])[0]
        self.uiCBColormaps.setCurrentIndex(cm_ind[0] if len(cm_ind) else 0)

        # add spectral to RGB conversion options
        self.uiCBSpecCMFs = _add_widget(width, QComboBox, None, 'activated', self._callback_combobox, self.spec_cmf_names)
        cmf_ind = np.where([name == self.spec_cmf_selected_name for name in self.spec_cmf_names])[0]
        self.uiCBSpecCMFs.setCurrentIndex(cmf_ind[0] if len(cmf_ind) else 0)
        self.uiCBSpecIlluminants = _add_widget(width, QComboBox, None, 'activated', self._callback_combobox, self.spec_illuminant_names)
        illum_ind = np.where([name == self.spec_illuminant_selected_name for name in self.spec_illuminant_names])[0]
        self.uiCBSpecIlluminants.setCurrentIndex(illum_ind[0] if len(illum_ind) else 0)
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
            _add_row(QLabel('collage:'), _multicolumn(self.uiCBCollageActive, self.uiCBCollageTight))
            _add_row(QLabel('per img:'), _multicolumn(self.uiCBCollageTranspose, self.uiCBCollageTransposeIms))
            _add_row(QLabel('NR x NC:'), _multicolumn(self.uiLECollageNr, self.uiLECollageNc))
            _add_row(QLabel('BW x BV:'), _multicolumn(self.uiLECollageBW, self.uiLECollageBV))
            _add_row(_hdiv())
        _add_row(QLabel('alpha:'), _multicolumn(self.uiCBHasAlpha, self.uiCBBlendAlpha))
        _add_row(_hdiv())
        _add_row(QLabel('crop:'), _multicolumn(self.uiCBCrop, self.uiCBCropGlobal))
        _add_row(QLabel('crop bgrnd:'), _multicolumn(self.uiLECropBackground, self.uiLEBackground))
        _add_row(_hdiv())
        _add_row(QLabel('annotate:'), _multicolumn(self.uiCBAnnotate, self.uiCBAnnotateNumbers))
        _add_row(QLabel('font size:'), self.uiLEFontSize)
        _add_row(QLabel('font value:'), self.uiLEFontColor)
        _add_row(_hdiv())
        _add_row(QLabel('info:'), self.uiLabelInfo)
        _add_row(_hdiv())

        _add_row(QLabel('cmap:'), self.uiCBColormaps)
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
        tmp = ui.text()
        try:
            try:
                tmp = json.loads(tmp)
                if isinstance(tmp, list):
                    tmp = np.array(tmp)
            except json.decoder.JSONDecodeError:
                try:
                    tmp = float(tmp)
                except:
                    return
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
            if self.nims > self.collage_nc * self.collage_nr:
                # increase nc to match selected nr given nims
                self.collage_nc = int(np.ceil(self.nims / self.collage_nr))
                self.uiLECollageNc.blockSignals(True)
                self.uiLECollageNr.blockSignals(True)
                self.uiLECollageNc.setText(str(self.collage_nc))
                self.uiLECollageNr.blockSignals(False)
                self.uiLECollageNc.blockSignals(False)
            self._display_collage()
        elif hasattr(self, 'uiLECollageNc') and ui == self.uiLECollageNc:
            self.collage_nc = int(tmp)
            if self.nims > self.collage_nc * self.collage_nr:
                # increase nr to match selected nc given nims
                self.collage_nr = int(np.ceil(self.nims / self.collage_nc))
                self.uiLECollageNc.blockSignals(True)
                self.uiLECollageNr.blockSignals(True)
                self.uiLECollageNr.setText(str(self.collage_nr))
                self.uiLECollageNr.blockSignals(False)
                self.uiLECollageNc.blockSignals(False)
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
        elif ui == self.uiLEBackground:
            self.background = tmp
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
        elif hasattr(self, 'uiCBCollageTight') and ui == self.uiCBCollageTight:
            self.collage_tight = bool(state)
            self._display_image()
        elif hasattr(self, 'uiCBCollageTranspose') and ui == self.uiCBCollageTranspose:
            self.collageTranspose = bool(state)
            self._display_image()
        elif hasattr(self, 'uiCBCollageTransposeIms') and ui == self.uiCBCollageTransposeIms:
            self.collageTransposeIms = bool(state)
            self._display_image()
        elif ui == self.uiCBHasAlpha:
            self.has_alpha = bool(state)
            self._display_image()
        elif ui == self.uiCBBlendAlpha:
            self.blend_alpha = bool(state)
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
        elif ui == self.uiCBColormaps:
            self.cm_name_selected = self.uiCBColormaps.currentText()
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
        if im.dtype != np.float32:
            im = im.astype(np.float32)
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
        if upper == lower:
            # enforce scaling by 1 to avoid zero division
            lower -= 0.5
            upper += 0.5
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
            # reset to default in case nc * nr < nims
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
        
        # pad array so it matches the product nc * nr
        ims = self.get_imgs(tonemap=True, decorate=True)
        coll = collage(images=ims,
                       nc=self.collage_nc,
                       nr=self.collage_nr,
                       tight=self.collage_tight,
                       transpose=self.collageTranspose,
                       transpose_ims=self.collageTransposeIms,
                       bv=self.collage_border_value,
                       bw=self.collage_border_width)

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
        im = self.ih.get_array()
        lims = np.array(self.ih.axes.axis())
        lims[0] = np.maximum(0, lims[0])
        lims[1] = np.minimum(im.shape[0], lims[1])
        lims[2] = np.minimum(im.shape[1], lims[2])
        lims[3] = np.maximum(0, lims[3])
        inds = (np.array(lims) + 0.5).astype(np.int32)
        xs = np.r_[inds[0]: inds[1]]
        ys = np.r_[inds[3]: inds[2]]
        
        for th in self.overlay_ths:
            try:
                th.remove()
            except:
                pass
        ths = []
        rgb2lum = np.r_[0.299, 0.587, 0.114]
        for xi, x0 in enumerate(xs):
            for yi, y0 in enumerate(ys):
                pixel = im[yi, xi, :]
                if np.sum(rgb2lum * pixel) > 0.5:
                    color = np.r_[0., 0., 0.]
                else:
                    color = np.r_[1., 1., 1.]
                if im.ndim == 3 and im.shape[2] == 3:
                    ths.append(self.ax.text(x0 - 0.5, y0 + 0.5, '% 6.3f\n% 6.3f\n% 6.3f\n' % tuple(pixel), fontsize=8, color=color))
                else:
                    ths.append(self.ax.text(x0 - 0.5, y0 + 0.5, '% 6.3f' % im[0, 0], fontsize=8, color=color))
        self.overlay_ths = ths
        self.fig.canvas.draw()

    def blend(self, im, alpha):
        """perform alpha blending of input image and some user-specified background"""
        bgrnd = np.array(self.background)
        while bgrnd.ndim < 3:
            bgrnd = bgrnd[None]
        if im.shape[2] == 1 and bgrnd.shape[2] > im.shape[2]:
            # handle intensity image with RGB background
            im = np.repeat(im, bgrnd.shape[2], axis=2)
        return alpha * im + (1 - alpha) * bgrnd

    def tonemap(self, im):
        """apply simple scaling & gamma based tonemapping to HDR image, convert spectral to RGB"""
        # TODO: add color mapping for single channel images
        if isinstance(im, np.matrix):
            im = np.array(im)

        if im.shape[2] == 1:
            # L
            if self.cm_name_selected == 'gray':
                im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 2 and self.has_alpha and self.blend_alpha:
            # LA
            im = self.blend(im[:, :, 0:1], im[:, :, 1:2])
            if im.shape[2] == 1:
                im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 2 and self.has_alpha and not self.blend_alpha:
            # discard A from LA
            im = np.repeat(im[:, :, 0:1], 3, axis=2)
        elif im.shape[2] == 2 and not self.has_alpha:
            # RG -> RGB
            im = np.concatenate((im, np.zeros((im.shape[0], im.shape[1], 1), dtype=im.dtype)), axis=2)
        elif im.shape[2] == 3:
            # RGB
            pass
        elif im.shape[2] == 4 and self.has_alpha and self.blend_alhpa:
            # RGBA
            im = self.blend(im[:, :, :3], im[:, :, 3:4])
        elif im.shape[2] == 4 and self.has_alpha and not self.blend_alhpa:
            # discard A from RGBA
            im = im[:, :, :3]
        elif im.shape[2] != 3:
            # project spectral to RGB
            if colour is None:
                raise NotImplemented('please install the colour-science package (pip install colour-science)')

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
        im = np.clip((im - self.offset) * self.scale, 0, 1) ** (1. / self.gamma)
        if self.cm_name_selected != 'gray':
            return cm.get_cmap(self.cm_name_selected)(im[..., 0])[..., :3]
        else:
            return im

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

    def _update_info(self, pixel=None):
        if pixel is not None:
            if len(pixel['value']) <= 3:
                # short pixel vectors are split at spaces and broken with newlines
                tmp = str(pixel['value'])
                while '  ' in tmp:
                    tmp = tmp.replace('  ', ' ')
                tmp = tmp.replace('[ ', '[').replace(' ]', ']')
                tmp = '\n'.join(tmp.split(' '))
            else:
                # longer (probably spectral ones) are broken at fixed lengths
                lines = ['']
                for ind, val in enumerate(pixel['value']):
                    last = ind == len(pixel['value']) - 1
                    if len(lines) == 1:
                        max_len = 21
                    else:
                        max_len = 28
                    if len(lines[-1]) < max_len:
                        lines[-1] += ('%.2f' if last else '%.2f, ') % val
                    else:
                        lines.append(('%.2f' if last else '%.2f, ') % val)
                tmp = '[' + '\n'.join(lines) + ']'
            pixel = '\n(%d,%d): %s' % (pixel['x'], pixel['y'], tmp)
        else:
            pixel = ' \n \n '
        size = str(self.get_img().shape)
        size = size.replace(', ', 'x')
        self.uiLabelInfo.setText('img: %d/%d, %s%s' % (self.imind + 1, self.nims, size, pixel))

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
        elif event.inaxes:
            im = self.get_img(tonemap=False, decorate=False)
            x = np.maximum(0, np.minimum(im.shape[1] - 1, int(event.xdata + 0.5)))
            y = np.maximum(0, np.minimum(im.shape[0] - 1, int(event.ydata + 0.5)))
            pixel = {'value': im[y, x], 'x': x, 'y': y}
            self._update_info(pixel)

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

    def _get_image_pos_canvas(self):
        extent = self.ax.get_window_extent()
        # canvas dimensions in pixels
        width_canvas, height_canvas = extent.x1 - extent.x0, extent.y1 - extent.y0

        # axis coordinates
        ax0, ax1 = self.ax.get_xlim()
        ay0, ay1 = self.ax.get_ylim()
        ix0, ix1, iy1, iy0 = self.ih.get_extent()

        # relative coordinates of image corners
        fx0 = (ix0 - ax0) / (ax1 - ax0)
        fx1 = 1 - (ax1 - ix1) / (ax1 - ax0)
        fy0 = 1 - (iy0 - ay0) / (ay1 - ay0)
        fy1 = (ay1 - iy1) / (ay1 - ay0)
        x0 = int(np.clip(np.round(width_canvas * fx0), 0, width_canvas))
        x1 = int(np.clip(np.round(width_canvas * fx1), 0, width_canvas))
        y0 = int(np.clip(np.round(height_canvas * fy0), 0, height_canvas))
        y1 = int(np.clip(np.round(height_canvas * fy1), 0, height_canvas))
        return x0, x1, y0, y1

    def copy_to_clipboard_zoomed(self):
        """get crop of zoomed in / out image on canvas at actual display resolution"""
        self.repaint()
        self.canvas.draw()

        x0, x1, y0, y1 = self._get_image_pos_canvas()
        im = QImage(self.canvas.grab())
        im = im.copy(x0, y0, x1 - x0, y1 - y0)
        im = qimage_to_np(im)

        # prevent garbage collection by storing the objects in the class
        self.clipboard_image = im
        self.clipboard_qimage = QImage(im, im.shape[1], im.shape[0], QImage.Format_ARGB32)
        if self.clipboard is None:
            self.clipboard = QApplication.clipboard()
        self.clipboard.setImage(self.clipboard_qimage)

    def save(self, ofname=None, zoomed=False, canvas=False, animation=False, tonemapped=True):
        if not tonemapped and (canvas or animation):
            warn('images / animations can only be written in tonemapped form when exporting the visible canvas')
            return

        try:
            if ofname is None:
                dialog = QFileDialog()
                ofname = dialog.getSaveFileName(parent=self,
                                                caption='file save path',
                                                directory=os.path.split(self.ofname)[0])[0]
            if ofname is None or not len(ofname):
                return
            self.ofname = ofname
            if os.path.splitext(ofname)[1].lower() in ['.gif', '.webp', '.mp4']:
                animation = True
            if zoomed:
                # export crop of current image that is determined by the zoom and pan level
                if tonemapped:
                    image = np.array(self.ih.get_array())
                else:
                    image = self.get_img()
                h, w = image.shape[:2]
                limits = self.ax.axis()
                x0 = np.max([0, int(limits[0] + 0.5)])
                x1 = np.min([w, int(limits[1] + 0.5)])
                y0 = np.max([0, int(limits[3] + 0.5)])
                y1 = np.min([h, int(limits[2] + 0.5)])
                image = image[y0:y1, x0:x1, :]
            elif canvas:
                # get only image content, not the white stuff from the canvas
                x0, x1, y0, y1 = self._get_image_pos_canvas()
                image = QImage(self.canvas.grab())
                image = image.copy(x0, y0, x1 - x0, y1 - y0)
                image = qimage_to_np(image)[:, :, -2::-1]
            elif animation:
                from pysmtb.utils import write_video
                ims = self.get_imgs(tonemap=True, decorate=False)
                # TODO: apply zoomed / canvas flags here, i.e. crop and / or scale each image
                if os.path.splitext(ofname)[1].lower() not in ['.webp', '.mp4', '.gif']:
                    print('unexpected file extension: %s' % os.path.splitext(ofname)[1].lower())
                else:
                    write_video(filename=ofname, frames=ims)
                return
            else:
                if tonemapped:
                    image = np.array(self.ih.get_array())
                else:
                    image = self.get_img()
            if not tonemapped and os.path.splitext(ofname)[1].lower() == '.exr':
                # export untonemapped images as OpenEXR
                from pysmtb.image import write_openexr
                write_openexr(ofname, image=image)
            else:
                # write any other formats
                imageio.imwrite(ofname, image)
        except Exception as e:
            warn(str(e))


if __name__ == '__main__':
    """
    basic command line interface, usage:
    
    python iv.py -i image.exr
    python iv.py -i image.exr --autoscale 0 --scale 2
    python iv.py -i image1.exr image2.exr --autoscale 0 --scale 2 --collage 1
    python iv.py -i *.exr --autoscale 1 --autoscaleGlobal 1 --collage 1 --nr 5 --nc 7
    """

    from argparse import ArgumentParser
    import glob
    import imageio
    from tqdm import tqdm

    from pysmtb.image import read_openexr

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+', default=['*'])
    args, unknown_args = parser.parse_known_args()

    iv_args = {}
    unknown_args = unknown_args[::-1]
    while len(unknown_args):
        arg = unknown_args.pop()
        if len(unknown_args) and arg.startswith('--'):
            val = unknown_args.pop()
            for val_type in [int, float, bool, str]:
                try:
                    val = val_type(val)
                    break
                except:
                    continue
            iv_args[arg[2:]] = val
    print(iv_args)

    inp = args.input
    if len(inp) == 1 and '*' in inp:
        filenames = sorted(glob.glob(inp))
    else:
        filenames = inp

    images = []
    for fn in tqdm(filenames, 'loading images'):
        ext = os.path.splitext(fn)[1].lower()
        if ext == '.exr':
            image, channels = read_openexr(fn, sort_rgb=True)
            rgb_inds = [ind for ind, c in enumerate(channels) if c.lower() in ['r', 'g', 'b']]
            luminance_ind = np.where(np.logical_or(np.array(channels) == 'L', np.array(channels) == 'l'))[0]
            if len(rgb_inds) > 0:
                image = image[:, :, np.array(rgb_inds)]
            elif len(luminance_ind):
                image = image[:, :, luminance_ind[0:1]]
            elif len(channels) > 3:
                chs = []
                # handle multispectral channels
                for ind, ch in enumerate(channels):
                    # if we can convert a channel name to float, it is likely a wavelength
                    try:
                        c = float(ch)
                        chs.append(ind)
                    except:
                        continue
                chs = np.array(chs)
                image = image[:, :, chs]
                # TODO: iv currently doesn't support specifying wavelength channels per image
            elif image.shape[-1] != 1:
                raise Exception('could not load image %s, unexpected channel count' % fn)
        elif ext in ['png', 'jpg', 'jpeg']:
            image = imageio.imread(fn)
        else:
            warning('unsupported file extension: ' + fn)
            continue

        images.append(image)
    v = iv(images, **iv_args)
    print('press any key to close the session')
    input()
