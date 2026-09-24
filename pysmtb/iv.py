"""
Created on Thu Oct 18 19:24:05 2018

@author: Sebastian Merzbach

interactive HDR and spectral image viewer: applies automatic tonemapping and cropping, creates multi image arrangements

setup:
virtualenv venv_pysmtb
. venv_pysmtb/bin/activate
pip install pysmtb

or manually install:
  click
  colour-science
  imageio
  matplotlib
  numpy
  OpenEXR
  PySide6
  pysmtb

On a Ubuntu, you might have to first run:
sudo apt-get install libopenexr-dev openexr zlib1g-dev python3-virtualenv

usage from code:

from pysmtb.iv import iv
v = iv(image1, image2, ...)
v = iv([image1, image2, image3], image4, image5, ...)
v = iv(images)  # image being H x W x C x N np.ndarray or torch.Tensor
v = iv(..., autoscale=True, autoscaleGlobal=True)
v = iv(..., autoscale=True, autoscaleGlobal=True, collage=True)
v = iv(..., dark=False)  # dark theme is on by default

usage from command line:

    iv image.exr
    iv image.exr --autoscale --scale 2
    iv image1.exr image2.exr --autoscale --scale 2 --collage
    iv *.exr --autoscale --autoscale-global 1 --collage --collage-nr 5 --collage-nc 7
    iv image.exr --no-dark


TODO: iv currently doesn't support specifying wavelength channels per image

"""

import click
from copy import deepcopy
from datetime import datetime
from functools import wraps
import math
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
from typing import Union, List, Tuple
from warnings import warn

# avoid problems on QT initialization; matplotlib must see PySide6, not PyQt5
os.environ['QT_STYLE_OVERRIDE'] = ''
os.environ['QT_API'] = 'pyside6'

from PySide6 import QtGui
import PySide6.QtCore as QtCore
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QFormLayout, QFrame, QGridLayout, QHBoxLayout, QLabel, \
    QLineEdit, QMainWindow, QPushButton, QSizePolicy, QSpacerItem, QSplitter, QStyleFactory, QVBoxLayout, \
    QWidget, QFileDialog

import matplotlib
try:
    matplotlib.use('QtAgg')
except:
    pass
import matplotlib.cm as cm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.transforms import Bbox

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import colour
except ModuleNotFoundError:
    colour = None

from pysmtb.image import crop_bounds, collage, qimage_to_np


def _qt_int(value):
    return value.value if hasattr(value, 'value') else int(value)


def _install_pyside_debug_hook():
    """keep the window alive at a debugpy breakpoint

    debugpy pumps events itself while paused, and that pump only knows PyQt.
    PySide6 already handles python -i and pdb via PyOS_InputHook.
    """
    try:
        from pydev_ipython.inputhook import set_inputhook
    except Exception:
        return

    def _hook():
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        return 0

    try:
        set_inputhook(_hook)
    except Exception:
        pass


'''
def MyPyQtSlot(*args):
    if len(args) == 0 or isinstance(args[0], types.FunctionType):
        args = []
    @QtCore.Slot(*args)
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

if hasattr(cm, '_cmap_registry'):
    cmap_names = list(cm._cmap_registry)
else:
    cmap_names = list(cm.cmaps_listed) + list(cm.datad)

@click.command()
@click.argument('filenames', nargs=-1)
@click.option('-s', '--scale', type=float, default=1.)
@click.option('-o', '--offset', type=float, default=0.)
@click.option('-g', '--gamma', type=float, default=1.)
@click.option('-c', '--colormap', type=click.Choice(cmap_names, case_sensitive=False), default='gray')
@click.option('--spec-wl0', type=float, default=380.0)
@click.option('--spec-wl1', type=float, default=730.0)
@click.option('--autoscale/--no-autoscale', default=True)
@click.option('--autoscale_global', is_flag=True, help='one range over all images concatenated')
@click.option('--autoscale-per-image', is_flag=True, help='normalize each image, then apply UI scale, offset and gamma')
@click.option('--autoscale-use-percentile/--no-autoscale-use-percentile', default=True)
@click.option('--autoscale-percentile', type=float, default=0.1)
@click.option('--collage', is_flag=True)
@click.option('--collage-tight/--no-collage-tight', default=True)
@click.option('--collage-transpose', is_flag=True)
@click.option('--collage-transpose-images', is_flag=True)
@click.option('--collage-nr', type=int, default=None)
@click.option('--collage-nc', type=int, default=None)
@click.option('--collage-border-width', type=int, default=0)
@click.option('--collage-border-value', type=float, default=0.0)
@click.option('--has-alpha/--no-has-alpha', default=True)
@click.option('--blend-alpha/--no-blend-alpha', default=True)
@click.option('--background', type=float, default=0.0)
@click.option('--crop', is_flag=True)
@click.option('--crop-left', type=int, default=None)
@click.option('--crop-right', type=int, default=None)
@click.option('--crop-top', type=int, default=None)
@click.option('--crop-bottom', type=int, default=None)
@click.option('--crop-width', type=int, default=None)
@click.option('--crop-height', type=int, default=None)
@click.option('--crop-stride-x', type=int, default=1)
@click.option('--crop-stride-y', type=int, default=1)
@click.option('--crop-global', is_flag=True)
@click.option('--crop-background', type=float, default=0.0)
@click.option('--annotate', is_flag=True)
@click.option('--annotate-numbers', is_flag=True)
@click.option('--font-size', type=int, default=12)
@click.option('--font-color', type=float, default=1.)
@click.option('--dark/--no-dark', default=True)
@click.option('-l', '--label', 'labels', type=str, default=None, multiple=True)
# non-IV options:
@click.option('-s', '--subsample', 'subsample', type=int, default=1)
@click.option('-r', '--roi', 'roi', type=str, default=None)
def iv_cli(filenames, **kwargs):
    """
    basic command line interface, usage:

    iv image.exr
    iv image.exr --no-autoscale --scale 2
    iv image1.exr image2.exr --no-autoscale --scale 2 --collage
    iv *.exr --autoscale --autoscale-global --collage --collage-nr 5 --collage-nc 7
    """

    import glob
    import imageio
    from tqdm import tqdm
    from pysmtb.image import read_openexr

    # inp = args.input
    if len(filenames) == 1 and '*' in filenames:
        filenames = sorted(glob.glob(filenames))

    # subsampling
    ss = kwargs['subsample']

    # cropping to ROI
    roi = kwargs['roi']
    if roi is not None:
        roi = [int(x) for x in roi.split(',')]
        x0, y0, width, height = roi

    # iterate over provided filenames and load images
    images = []
    labels = []
    for fn in tqdm(filenames, 'loading images'):
        ext = os.path.splitext(fn)[1].lower()
        if ext == '.exr':
            # special treatment for EXR images
            image, channels = read_openexr(fn, sort_rgb=True)
            rgb_inds = [ind for ind, c in enumerate(channels) if c.lower() in ['r', 'g', 'b']]
            luminance_ind = np.where(np.logical_or(np.array(channels) == 'L', np.array(channels) == 'l'))[0]
            if len(rgb_inds) > 0:
                image = image[:, :, np.array(rgb_inds)]
            elif len(luminance_ind):
                image = image[:, :, luminance_ind[0:1]]
            elif len(channels) == 3:
                # default treatment of 3-channel images as RGB
                pass
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
        else:
            try:
                image = imageio.imread(fn)
            except Exception as ex:
                warn('Warning: could not read image file ' + fn + ', caught exception:\n' + str(ex))
                continue

        if roi is not None:
            image = image[y0:y0+height:ss, x0:x0+width:ss]
        else:
            image = image[::ss, ::ss]
        images.append(image)
        labels.append(fn)

    # only set file name labels if no other labels were provided
    if kwargs['labels'] is None or len(kwargs['labels']) == 0:
        kwargs['labels'] = labels

    if not len(images):
        sys.exit('No images loaded.')

    v = IV(images, **kwargs)
    print('Press any key to close the session:')
    input()


def iv(*args, **kwargs):
    return IV(*args, **kwargs)


def dragfloat_value(origin, dx, speed, factor=1.0, logarithmic=False, relative=False, minimum=None, maximum=None):
    """value after an ImGui-style drag

    dx is pixels from the press position. factor is 0.01 with alt, 10 with shift.
    logarithmic applies the delta in log space. relative scales a linear delta by max(|origin|, 1).
    """
    if logarithmic:
        v0 = origin if origin != 0 else 1e-8
        sign = 1.0 if v0 > 0 else -1.0
        value = sign * math.exp(math.log(abs(v0)) + dx * speed * factor)
        if not math.isfinite(value):
            value = math.copysign(1e12, sign)
    else:
        magnitude = max(abs(origin), 1.0) if relative else 1.0
        value = origin + dx * speed * factor * magnitude
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


class DragFloat(QWidget):
    """slider-like float editor, same gestures as ImGui DragFloat

    drag horizontally to change the value, ctrl+click to type one,
    hold shift to speed up (x10) and alt to slow down (x0.01)
    """
    valueChanged = QtCore.Signal(float)

    def __init__(self, value=0.0, speed=None, minimum=None, maximum=None,
                 logarithmic=False, relative=False, parent=None):
        super().__init__(parent)
        if speed is None:
            if minimum is not None and maximum is not None and maximum > minimum:
                speed = (maximum - minimum) * 0.01
            else:
                speed = 0.01
        self.speed = speed
        self.minimum = minimum
        self.maximum = maximum
        self.logarithmic = logarithmic
        self.relative = relative
        self.value = float(value)
        self._clamp()
        self._dragging = False
        self._editing = False
        self._press_x = 0
        self._press_value = self.value
        self.setMinimumHeight(22)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.SizeHorCursor)
        self.setToolTip('drag to adjust    shift: faster    alt: slower    ctrl+click: edit')
        self._edit = QLineEdit(self)
        self._edit.setAlignment(Qt.AlignCenter)
        self._edit.setFrame(False)
        self._edit.hide()
        self._edit.installEventFilter(self)
        self._edit.editingFinished.connect(self._commit_edit)

    def sizeHint(self):
        return QtCore.QSize(120, 24)

    def setValue(self, value):
        self.value = float(value)
        self._clamp()
        self.update()

    def _clamp(self):
        if self.minimum is not None:
            self.value = max(self.minimum, self.value)
        if self.maximum is not None:
            self.value = min(self.maximum, self.value)

    def _speed_factor(self):
        mods = QApplication.queryKeyboardModifiers()
        factor = 1.0
        if mods & Qt.AltModifier:
            factor *= 0.01
        if mods & Qt.ShiftModifier:
            factor *= 10.0
        return factor

    def _emit_user(self, value):
        old = self.value
        self.setValue(value)
        if self.value != old:
            self.valueChanged.emit(self.value)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self._editing:
            return
        if event.modifiers() & Qt.ControlModifier:
            self._begin_edit()
            return
        self._dragging = True
        self._press_x = event.x()
        self._press_value = self.value
        self.grabMouse()

    def mouseMoveEvent(self, event):
        if not self._dragging:
            return
        value = dragfloat_value(
            self._press_value, event.x() - self._press_x, self.speed, self._speed_factor(),
            logarithmic=self.logarithmic, relative=self.relative,
            minimum=self.minimum, maximum=self.maximum)
        self._emit_user(value)

    def mouseReleaseEvent(self, event):
        if self._dragging and event.button() == Qt.LeftButton:
            self._dragging = False
            self.releaseMouse()

    def _begin_edit(self):
        self._editing = True
        self._edit.setText(format(self.value, '.6g'))
        self._edit.setGeometry(self.rect().adjusted(1, 1, -1, -1))
        self._edit.show()
        self._edit.setFocus(Qt.OtherFocusReason)
        self._edit.selectAll()

    def _commit_edit(self):
        if not self._editing:
            return
        text = self._edit.text().strip()
        self._editing = False
        self._edit.hide()
        try:
            value = float(text)
        except ValueError:
            self.update()
            return
        self._emit_user(value)

    def eventFilter(self, obj, event):
        if obj is self._edit and event.type() == QtCore.QEvent.ShortcutOverride and event.key() == Qt.Key_Escape:
            event.accept()
            return True
        if obj is self._edit and event.type() == QtCore.QEvent.KeyPress and event.key() == Qt.Key_Escape:
            self._editing = False
            self._edit.hide()
            self.update()
            return True
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        self._edit.setGeometry(self.rect().adjusted(1, 1, -1, -1))
        super().resizeEvent(event)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = self.rect().adjusted(0, 0, -1, -1)
        palette = self.palette()
        painter.setPen(QtGui.QPen(palette.color(QtGui.QPalette.Mid)))
        painter.setBrush(palette.color(QtGui.QPalette.Base))
        painter.drawRoundedRect(rect, 3, 3)
        if self.minimum is not None and self.maximum is not None and self.maximum > self.minimum:
            t = (self.value - self.minimum) / (self.maximum - self.minimum)
            t = min(1.0, max(0.0, t))
            fill = QtCore.QRect(rect.adjusted(1, 1, -1, -1))
            fill.setWidth(max(0, int(round(fill.width() * t))))
            color = palette.color(QtGui.QPalette.Highlight)
            color.setAlpha(160)
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(fill, 2, 2)
        if not self._editing:
            painter.setPen(palette.color(QtGui.QPalette.Text))
            painter.drawText(rect, Qt.AlignCenter, format(self.value, '.6g'))


class IV(QMainWindow):
    """interactive HDR and spectral image viewer

    applies automatic tonemapping and cropping, creates multi image arrangements"""
    def __init__(self, *args,
                 scale: float = 1.,
                 offset: float = 0.,
                 gamma: float = 1.,
                 colormap: str = 'gray',
                 autoscale: bool = True,
                 autoscale_global: bool = False,
                 autoscale_per_image: bool = False,
                 autoscale_use_percentile: bool = True,
                 autoscale_percentile: float = 0.1,
                 collage: bool = False,
                 collage_tight: bool = True,
                 collage_transpose: bool = False,
                 collage_transpose_images: bool = False,
                 collage_nr: int = None,
                 collage_nc: int = None,
                 collage_border_width: int = 0,
                 collage_border_value: int = 0,
                 has_alpha: bool = True,
                 blend_alpha: bool = True,
                 background: float = 0.0,
                 crop: bool = False,
                 crop_left: int = None,
                 crop_right: int = None,
                 crop_top: int = None,
                 crop_bottom: int = None,
                 crop_width: int = None,
                 crop_height: int = None,
                 crop_stride_x: int = 1,
                 crop_stride_y: int = 1,
                 crop_global: bool = False,
                 crop_background: float = 0.0,
                 annotate: bool = False,
                 annotate_numbers: bool = True,
                 font_size: int = 12,
                 font_color: float = 1.0,
                 dark: bool = True,
                 labels: Union[List, Tuple] = (),
                 spec_wl0: float = 380.0,
                 spec_wl1: float = 730.0,
                 **kwargs):

        self.app = QtCore.QCoreApplication.instance()
        if self.app is None:
            self.app = QApplication([''])
        _install_pyside_debug_hook()
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
            if str(type(inp)) == "<class 'torch.Tensor'>":
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
        self.labels = list(labels)
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
        if self.nims == 0:
            return

        # tonemapping
        self.scale = scale
        self.gamma = gamma
        self.offset = offset
        self.autoscaleEnabled = autoscale
        # exclusive: current image, one range over the concatenation, or per-image then UI tonemap
        self.autoscalePerImage = bool(autoscale_per_image)
        self.autoscaleGlobal = bool(autoscale_global) and not self.autoscalePerImage
        self.image_offsets = None
        self.image_scales = None
        self.autoscaleLower = self.autoscaleEnabled
        self.autoscaleUpper = self.autoscaleEnabled
        self.autoscaleUsePrctiles = autoscale_use_percentile
        self.autoscalePrctiles = autoscale_percentile
        if np.isscalar(self.autoscalePrctiles):
            self.autoscalePrctiles = np.array([self.autoscalePrctiles, 100. - self.autoscalePrctiles])
        assert len(self.autoscalePrctiles) == 2, 'autoscalePrctiles must have 2 elements!'
        self.autoscalePrctiles[0] = np.clip(self.autoscalePrctiles[0], 0., 50.)
        self.autoscalePrctiles[1] = np.clip(self.autoscalePrctiles[1], 50., 100.)

        # colormapping for scalar-valued inputs
        self.cm_names = cmap_names
        self.cm_name_selected = colormap

        # collage
        self.collageActive = collage
        self.collage_tight = collage_tight
        self.collageTranspose = collage_transpose
        self.collageTransposeIms = collage_transpose_images
        if collage_nr is not None and collage_nc is not None:
            collage_nr = int(np.maximum(1, collage_nr))
            collage_nc = int(np.maximum(1, collage_nc))
        elif collage_nr is not None:
            collage_nr = int(np.maximum(1, collage_nr))
            collage_nc = int(np.ceil(self.nims / collage_nr))
        elif collage_nc is not None:
            collage_nc = int(np.maximum(1, collage_nc))
            collage_nr = int(np.ceil(self.nims / collage_nc))
        else:
            collage_nc = int(np.ceil(np.sqrt(self.nims)))
            collage_nr = int(np.ceil(self.nims / collage_nc))
        self.collage_nr = collage_nr
        self.collage_nc = collage_nc
        self.collage_border_width = collage_border_width
        self.collage_border_value = collage_border_value

        # alpha mapping
        self.has_alpha = has_alpha
        self.blend_alpha = blend_alpha
        self.background = background

        # automatic cropping
        self.crop = crop
        self.crop_global = crop_global
        self.crop_background = crop_background
        # explicit crop ROI
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.crop_stride_x = crop_stride_x
        self.crop_stride_y = crop_stride_y

        # image annotations
        self.annotate = annotate
        self.annotate_numbers = annotate_numbers
        self.font_size = font_size
        self.font_color = font_color
        self.dark = dark
        if len(self.labels) == 0:
            self.labels = None
        if self.labels is not None:
            assert len(self.labels) == len(self.images), 'number of labels %d must match number of images %d'\
                                                         % (len(self.labels), len(self.images))

        self.zoom_factor = 1.1
        self.x_zoom = True
        self.y_zoom = True
        self.x_stop_at_orig = True
        self.y_stop_at_orig = True

        # stores np.ndarray, QImage and QApplication.clipboard() objects to prevent garbage collection when copying
        # canvas or image to clipboard
        self.clipboard_image = None
        self.clipboard_qimage = None
        self.clipboard = None

        # spectral to RGB conversion stuff
        self.spec_wl0 = spec_wl0
        self.spec_wl1 = spec_wl1
        if colour is not None:
            self.spec_cmf_names = list(colour.MSDS_CMFS.keys())
            self.spec_illuminant_names = list(colour.SDS_ILLUMINANTS.keys())
        else:
            self.spec_cmf_names = ['pip install colour-science']
            self.spec_illuminant_names = ['pip install colour-science']
        self.spec_cmf_selected_name = 'CIE 1931 2 Degree Standard Observer'
        self.spec_illuminant_selected_name = 'E'

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
        if self.crop_left is None \
                or self.crop_top is None \
                or self.crop_right is None and self.crop_width is None \
                or self.crop_bottom is None and self.crop_height is None:
            res = crop_bounds(self.images, apply=False, crop_global=self.crop_global, background=self.crop_background)

        if self.crop_left is None:
            self.xmins = res['xmins']
        else:
            self.xmins = [self.crop_left] * self.nims

        if self.crop_right is None and self.crop_width is None:
            self.xmaxs = res['xmaxs']
        elif self.crop_right is not None:
            self.xmaxs = [self.crop_right] * self.nims
        elif self.crop_width is not None:
            self.xmaxs = [self.crop_left + self.crop_width] * self.nims

        if self.crop_top is None:
            self.ymins = res['ymins']
        else:
            self.ymins = [self.crop_top] * self.nims

        if self.crop_bottom is None and self.crop_height is None:
            self.ymaxs = res['ymaxs']
        elif self.crop_bottom is not None:
            self.ymaxs = [self.crop_bottom] * self.nims
        elif self.crop_height is not None:
            self.ymaxs = [self.crop_top + self.crop_height] * self.nims

    def _apply_dark_theme(self):
        style = QStyleFactory.create('Fusion')
        if style is not None:
            self.setStyle(style)
        bg = QtGui.QColor(45, 45, 48)
        base = QtGui.QColor(30, 30, 30)
        button = QtGui.QColor(60, 60, 64)
        text = QtGui.QColor(220, 220, 220)
        disabled = QtGui.QColor(127, 127, 127)
        highlight = QtGui.QColor(61, 110, 158)
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, bg)
        palette.setColor(QtGui.QPalette.WindowText, text)
        palette.setColor(QtGui.QPalette.Base, base)
        palette.setColor(QtGui.QPalette.AlternateBase, button)
        palette.setColor(QtGui.QPalette.ToolTipBase, base)
        palette.setColor(QtGui.QPalette.ToolTipText, text)
        palette.setColor(QtGui.QPalette.Text, text)
        palette.setColor(QtGui.QPalette.Button, button)
        palette.setColor(QtGui.QPalette.ButtonText, text)
        palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 80, 80))
        palette.setColor(QtGui.QPalette.Link, highlight)
        palette.setColor(QtGui.QPalette.Highlight, highlight)
        palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))
        palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, disabled)
        palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, disabled)
        palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, disabled)
        app = QApplication.instance()
        if app is not None:
            if style is not None:
                app.setStyle(style)
            app.setPalette(palette)
        self.setPalette(palette)
        self.setAutoFillBackground(True)

    def _style_canvas(self):
        if not self.dark:
            return
        color = '#1e1e1e'
        self.fig.patch.set_facecolor(color)
        self.ax.set_facecolor(color)

    def _init_ui(self):
        if self.dark:
            self._apply_dark_theme()
        self.widget = QWidget()
        if self.dark:
            self.widget.setAutoFillBackground(True)

        self.fig = Figure(dpi=100, facecolor='#1e1e1e' if self.dark else 'white')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.widget)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_position(Bbox([[0, 0], [1, 1]]))
        self.ax.set_aspect(1, 'datalim')
        self.ax.set_anchor('NW')
        self.ax.set_clip_on(False)
        self.ax.set_axis_off()
        self._style_canvas()
        self._invert_y()

        width = 200

        def _add_widget(w, widget, label, signal=None, callback=None, value=None):
            widget = widget(None if widget == QComboBox else str(value) if label is None else str(label))
            widget.setMaximumWidth(int(w))
            if label is not None and value is not None:
                if isinstance(widget, QCheckBox):
                    widget.setTristate(False)
                    widget.setCheckState(Qt.Checked if value else Qt.Unchecked)
            if isinstance(widget, QComboBox):
                widget.addItems(value)
            if callback is not None:
                getattr(widget, signal).connect(lambda *args: callback(widget, *args))
            return widget

        self.uiLabelModifiers = QLabel('')
        self.uiLabelModifiers.setMaximumWidth(int(width))
        self.uiLEScale = DragFloat(self.scale, speed=0.01, logarithmic=True)
        self.uiLEGamma = DragFloat(self.gamma, speed=0.01, logarithmic=True)
        self.uiLEOffset = DragFloat(self.offset, speed=0.01, relative=True)
        for widget in (self.uiLEScale, self.uiLEGamma, self.uiLEOffset):
            widget.setMaximumWidth(int(width))
        self.uiLEScale.valueChanged.connect(self.set_scale)
        self.uiLEGamma.valueChanged.connect(self.set_gamma)
        self.uiLEOffset.valueChanged.connect(self.set_offset)
        self.uiCBAutoscaleLower = _add_widget(width // 2, QCheckBox, 'lower', 'stateChanged', self._callback_check_box, self.autoscaleLower)
        self.uiCBAutoscaleUpper = _add_widget(width // 2, QCheckBox, 'upper', 'stateChanged', self._callback_check_box, self.autoscaleUpper)
        self.uiLEAutoscalePrctileLower = DragFloat(self.autoscalePrctiles[0], minimum=0., maximum=50.)
        self.uiLEAutoscalePrctileUpper = DragFloat(self.autoscalePrctiles[1], minimum=50., maximum=100.)
        self.uiLEAutoscalePrctileLower.setMaximumWidth(width // 2)
        self.uiLEAutoscalePrctileUpper.setMaximumWidth(width // 2)
        self.uiLEAutoscalePrctileLower.valueChanged.connect(self._set_autoscale_prctile_lower)
        self.uiLEAutoscalePrctileUpper.valueChanged.connect(self._set_autoscale_prctile_upper)
        self.uiLabelAutoscaleLower = _add_widget(width // 2, QLabel, '%f' % 0.)
        self.uiLabelAutoscaleUpper = _add_widget(width // 2, QLabel, '%f' % 1.)
        self.uiCBAutoscaleUsePrctiles = _add_widget(width // 2, QCheckBox, 'prcntiles', 'stateChanged', self._callback_check_box, self.autoscaleUsePrctiles)
        self.uiCBAutoscaleGlobal = _add_widget(width // 2, QCheckBox, 'global', 'stateChanged', self._callback_check_box, False)
        self.uiCBAutoscaleGlobal.blockSignals(True)
        self.uiCBAutoscaleGlobal.setTristate(True)
        if self.autoscalePerImage:
            self.uiCBAutoscaleGlobal.setCheckState(Qt.PartiallyChecked)
            self.uiCBAutoscaleGlobal.setText('individually')
        elif self.autoscaleGlobal:
            self.uiCBAutoscaleGlobal.setCheckState(Qt.Checked)
            self.uiCBAutoscaleGlobal.setText('jointly')
        else:
            self.uiCBAutoscaleGlobal.setCheckState(Qt.Unchecked)
        self.uiCBAutoscaleGlobal.blockSignals(False)
        self.uiCBAutoscaleGlobal.setMaximumWidth(width)
        self.uiCBAutoscaleGlobal.setToolTip(
            'global: current image\n'
            'jointly: one range over every image concatenated\n'
            'individually: normalize each image, then apply UI scale / offset / gamma')
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
        self.uiCBColormaps = _add_widget(width // 2, QComboBox, None, 'activated', self._callback_combobox, self.cm_names)
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
        self.uiLabelInfo.setMaximumWidth(int(width))
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
        vbox.addItem(QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding))
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

    def _set_autoscale_prctile_lower(self, value):
        self.autoscalePrctiles[0] = value
        self.autoscale()

    def _set_autoscale_prctile_upper(self, value):
        self.autoscalePrctiles[1] = value
        self.autoscale()

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
        
        if hasattr(self, 'uiLECollageNr') and ui == self.uiLECollageNr:
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
        state = _qt_int(state)
        print(ui, state)
        if ui == self.uiCBAutoscaleUsePrctiles:
            self.autoscaleUsePrctiles = bool(state)
            if self.autoscaleEnabled:
                self.autoscale()
        elif ui == self.uiCBAutoscaleGlobal:
            self._set_autoscale_scope(state, reset_final=True)
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
            im = im[self.ymins[i]:self.ymaxs[i]:self.crop_stride_y, self.xmins[i]:self.xmaxs[i]:self.crop_stride_x, :]
        if im.dtype != np.float32:
            im = im.astype(np.float32)
        if tonemap:
            im = self.tonemap(im, image_index=i)
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
            from pysmtb.image import annotate_image
            if self.annotate_numbers:
                label += str(i) + ' '
            if self.labels is not None:
                label += self.labels[i]
            if im.shape[2] == 3:
                im = annotate_image(im, label, font_size=self.font_size, font_color=self.font_color)
            else:
                im = annotate_image(im[:, :, 0], label, font_size=self.font_size, font_color=self.font_color, stroke_color=np.clip(1.-self.font_color, 0, 1))
        return im
    
    def _value_range(self, values):
        values = np.asarray(values)
        if self.autoscaleUsePrctiles:
            lower, upper = np.percentile(values, self.autoscalePrctiles)
        else:
            lower = np.min(values)
            upper = np.max(values)
        lower = float(lower)
        upper = float(upper)
        if upper == lower:
            lower -= 0.5
            upper += 0.5
        return lower, upper

    def _concat_pixels(self):
        ims = self.get_imgs(tonemap=False, decorate=False)
        if len(ims) == 1:
            return np.ravel(ims[0])
        return np.concatenate([np.ravel(im) for im in ims])

    def _update_per_image_tonemap(self):
        lowers = []
        uppers = []
        for im in self.get_imgs(tonemap=False, decorate=False):
            lower, upper = self._value_range(im)
            lowers.append(lower)
            uppers.append(upper)
        self.image_offsets = np.asarray(lowers, dtype=np.float64)
        self.image_scales = 1.0 / np.maximum(np.asarray(uppers, dtype=np.float64) - self.image_offsets, 1e-12)

    def _set_autoscale_scope(self, state, reset_final=False, update_checkbox=False):
        state = _qt_int(state)
        entering_each = state == _qt_int(Qt.PartiallyChecked) and not self.autoscalePerImage
        self.autoscalePerImage = state == _qt_int(Qt.PartiallyChecked)
        self.autoscaleGlobal = state == _qt_int(Qt.Checked)
        self.uiCBAutoscaleGlobal.setText({0: 'global', 1: 'individually', 2: 'jointly'}.get(state, 'global'))
        if update_checkbox:
            self.uiCBAutoscaleGlobal.blockSignals(True)
            self.uiCBAutoscaleGlobal.setCheckState(Qt.CheckState(state))
            self.uiCBAutoscaleGlobal.blockSignals(False)
        # per-image normalization maps each image to 0..1, so the UI sliders become the shared grade
        if entering_each and reset_final:
            self.set_offset(0., False)
            self.set_scale(1., False)
        if self.autoscalePerImage or self.autoscaleEnabled:
            self.autoscale()
        else:
            self._display_image()

    def autoscale(self):
        """autoscale between user-selected percentiles"""
        if self.autoscalePerImage:
            self._update_per_image_tonemap()
            self._display_image()
            return
        if self.autoscaleGlobal:
            lower, upper = self._value_range(self._concat_pixels())
        else:
            lower, upper = self._value_range(self.get_img(tonemap=False, decorate=False))
        if self.autoscaleLower:
            self.set_offset(lower, False)
            self.uiLabelAutoscaleLower.setText('%f' % lower)
        if self.autoscaleUpper:
            self.set_scale(1. / ((upper - lower) if upper != lower else 1), True)
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
        self._style_canvas()
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
            self._style_canvas()
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

    def tonemap(self, im, image_index=None):
        """apply simple scaling & gamma based tonemapping to HDR image, convert spectral to RGB"""
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
        elif im.shape[2] == 4 and self.has_alpha and self.blend_alpha:
            # RGBA
            im = self.blend(im[:, :, :3], im[:, :, 3:4])
        elif im.shape[2] == 4 and self.has_alpha and not self.blend_alpha:
            # discard A from RGBA
            im = im[:, :, :3]
        elif im.shape[2] != 3:
            # project spectral to RGB
            if colour is None:
                raise NotImplemented('please install the colour-science package (pip install colour-science)')

            wl_range = self.spec_wl1 - self.spec_wl0
            spec_shape = colour.SpectralShape(self.spec_wl0, self.spec_wl1, wl_range / np.maximum(1, (im.shape[2] - 1)))

            illuminant = deepcopy(colour.SDS_ILLUMINANTS[self.spec_illuminant_selected_name])
            illuminant = illuminant.align(shape=spec_shape)
            cmfs = deepcopy(colour.MSDS_CMFS[self.spec_cmf_selected_name])
            cmfs = cmfs.align(shape=spec_shape)
            im = colour.msds_to_XYZ(im, cmfs, illuminant, method='Integration', shape=spec_shape)
            if self.spec_cmf_selected_name.lower().startswith('cie'):
                im = colour.XYZ_to_sRGB(im / 100)
            else:
                im /= 100
        if self.autoscalePerImage:
            if self.image_offsets is None:
                self._update_per_image_tonemap()
            idx = self.imind if image_index is None else image_index
            off = self.image_offsets[idx] if self.autoscaleLower else 0.
            sc = self.image_scales[idx] if self.autoscaleUpper else 1.
            im = np.clip((im - off) * sc, 0, 1)
        im = np.clip((im - self.offset) * self.scale, 0, 1) ** (1. / (self.gamma if self.gamma != 0 else 1.))
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
                self._style_canvas()
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
        self.uiLEScale.setValue(self.scale)
        self.uiLabelAutoscaleLower.setText('%f' % self.offset)
        self.uiLabelAutoscaleUpper.setText('%f' % ((1 / (self.scale if self.scale != 0 else 1)) + self.offset))
        if redraw:
            self._display_image()

    def set_gamma(self, gamma, redraw=True):
        self.gamma = gamma
        self.uiLEGamma.setValue(self.gamma)
        if redraw:
            self._display_image()
    
    def set_offset(self, offset, redraw=True):
        self.offset = offset
        self.uiLEOffset.setValue(self.offset)
        self.uiLabelAutoscaleLower.setText('%f' % self.offset)
        self.uiLabelAutoscaleUpper.setText('%f' % ((1 / (self.scale if self.scale != 0 else 1)) + self.offset))
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
            self.set_gamma(1., redraw=False)
        elif key == Qt.Key_L:
            # update axes for single image dimensions
            if self.collageActive:
                self._switch_to_single_image()
            else:
                # toggle showing collage
                self.collageActive = not self.collageActive
            # also clear per-image normalization; global becomes a single shared range
            self.autoscalePerImage = False
            self.autoscaleGlobal = not self.autoscaleGlobal
            self.uiCBAutoscaleGlobal.blockSignals(True)
            self.uiCBAutoscaleGlobal.setCheckState(Qt.Checked if self.autoscaleGlobal else Qt.Unchecked)
            self.uiCBAutoscaleGlobal.setText('jointly' if self.autoscaleGlobal else 'global')
            self.uiCBAutoscaleGlobal.blockSignals(False)
        elif key == Qt.Key_O:
            self.set_offset(0., redraw=False)
        elif key == Qt.Key_P:
            if not self.autoscaleGlobal and not self.autoscalePerImage:
                state = Qt.Checked
            elif self.autoscaleGlobal:
                state = Qt.PartiallyChecked
            else:
                state = Qt.Unchecked
            self._set_autoscale_scope(state, reset_final=True, update_checkbox=True)
            print('autoscale scope: %s' % self.uiCBAutoscaleGlobal.text())
            return
        elif key == Qt.Key_S:
            self.set_scale(1., redraw=False)
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
            self.uiLEAutoscalePrctileLower.setValue(self.autoscalePrctiles[0])
            self.uiLEAutoscalePrctileUpper.setValue(self.autoscalePrctiles[1])
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
        im = QImage(im.tobytes(), w, h, nc * w, QImage.Format_RGB888).copy()
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
        fx0 = (ix0 - ax0) / ((ax1 - ax0) if ax0 != ax1 else 1)
        fx1 = 1 - (ax1 - ix1) / ((ax1 - ax0) if ax0 != ax1 else 1)
        fy0 = 1 - (iy0 - ay0) / ((ay1 - ay0) if ay0 != ay1 else 1)
        fy1 = (ay1 - iy1) / ((ay1 - ay0) if ay0 != ay1 else 1)
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
        im = self.canvas.grab().toImage()
        im = im.copy(x0, y0, x1 - x0, y1 - y0)
        im = qimage_to_np(im)

        # prevent garbage collection by storing the objects in the class
        self.clipboard_image = np.ascontiguousarray(im)
        self.clipboard_qimage = QImage(
            self.clipboard_image,
            self.clipboard_image.shape[1],
            self.clipboard_image.shape[0],
            self.clipboard_image.strides[0],
            QImage.Format_ARGB32,
        ).copy()
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
                                                dir=os.path.split(self.ofname)[0])[0]
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
                image = self.canvas.grab().toImage()
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
