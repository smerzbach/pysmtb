"""
Created on Wed Jan 14 23:01:57 2020

@author: Sebastian Merzbach <smerzbach@gmail.com>
"""

import numpy as np
import os
import re
import sys
from typing import Dict

try:
    import torch
except:
    pass


def annotate_image(image, label, font_path=None, font_size=16, font_color=[1.], stroke_color=[0.], stroke_width=1,
                   x=0, y=0, overlay=False, overlay_color=1., overlay_bbox=None):
    from PIL import Image
    from PIL import ImageFont
    from PIL import ImageDraw

    if font_path is None:
        if sys.platform == 'win32':
            font_path = os.path.join(os.getenv('WINDIR'), 'Fonts', 'cour.ttf')
        else:
            font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'

    image = np.atleast_3d(image)

    try:
        len(font_color)
    except:
        font_color = [font_color]

    try:
        len(stroke_color)
    except:
        stroke_color = [stroke_color]

    font_color = font_color + [font_color[-1] for _ in range(image.shape[2] - len(font_color))]
    stroke_color = stroke_color + [stroke_color[-1] for _ in range(image.shape[2] - len(stroke_color))]
    font_color = tuple([int(c * 255) for c in font_color])
    stroke_color = tuple([int(c * 255) for c in stroke_color])

    mask = Image.fromarray(np.zeros(image.shape[:2] + (image.shape[2] + 1,), dtype=np.uint8))
    draw = ImageDraw.Draw(mask)
    font = ImageFont.truetype(font_path, font_size)

    if image.ndim == 3:
        draw.text((x, y), text=label, fill=tuple(font_color) + (255,), font=font, stroke_width=stroke_width, stroke_fill=tuple(stroke_color) + (255,))
    else:
        draw.text((x, y), text=label, fill=tuple(font_color) + (255,), font=font, stroke_width=stroke_width, stroke_fill=tuple(stroke_color) + (255,))
    mask = np.atleast_3d(np.array(mask, dtype=np.float) / 255.).astype(image.dtype)
    alpha = np.atleast_3d(mask[:,:,-1])
    mask = np.atleast_3d(mask[:,:,:-1])

    if overlay:
        if overlay_bbox is not None:
            x0 = overlay_bbox['x0']
            x1 = overlay_bbox['x1']
            y0 = overlay_bbox['y0']
            y1 = overlay_bbox['y1']
        else:
            h, w = alpha.shape[:2]
            ys, xs = np.where(alpha[:, :, 0] != 0)
            y0 = np.maximum(0, np.minimum(np.min(ys), y))
            y1 = np.minimum(h, np.maximum(np.max(ys), y))
            x0 = np.maximum(0, np.minimum(np.min(xs), x))
            x1 = np.minimum(w, np.maximum(np.max(xs), x))
        image[y0:y1, x0:x1, :] = overlay_color

    return (1 - alpha) * image + alpha * mask


def pad(image, new_width, new_height, new_num_channels=None, value=0., center=True):
    height, width = image.shape[:2]
    pad_width = new_width - width
    pad_height = new_height - height
    margins0 = [pad_height // 2, pad_height - pad_height // 2]
    margins1 = [pad_width // 2, pad_width - pad_width // 2]

    image = np.concatenate((value * np.ones((margins0[0],) + image.shape[1:4], dtype=image.dtype),
                            image,
                            value * np.ones((margins0[1],) + image.shape[1:4], dtype=image.dtype)), axis=0)
    image = np.concatenate((value * np.ones((image.shape[0], margins1[0]) + image.shape[2:4], dtype=image.dtype),
                            image,
                            value * np.ones((image.shape[0], margins1[1]) + image.shape[2:4], dtype=image.dtype)), axis=1)
    if not new_num_channels is None and image.shape[2] < new_num_channels:
        image = np.concatenate((image, np.atleast_3d(value * np.ones(image.shape[:2] + (new_num_channels - image.shape[2],), dtype=image.dtype))), axis=2)

    return image


def collage(images, **kwargs):
    if isinstance(images, np.ndarray):
        if images.ndim == 4:
            images = [images[:, :, :, i] for i in range(images.shape[3])]
        else:
            images = [images]
    if isinstance(images, list):
        images = [np.atleast_3d(im) for im in images]

    nims = len(images)

    nc = kwargs.get('nc', None)  # number of columns
    nr = kwargs.get('nr', None)  # number of rows
    if nc is None:
        nc = int(np.ceil(np.sqrt(nims)))
    if nr is None:
        nr = int(np.ceil(nims / nc))
    bw = kwargs.get('bw', 0)  # border width
    bv = kwargs.get('bv', 0)  # border value
    transpose = kwargs.get('transpose', False)
    transposeIms = kwargs.get('transposeIms', False)
    fill_value = kwargs.get('fill_value', 1)

    if nr * nc < nims:
        nc = int(np.ceil(np.sqrt(nims)))
        nr = int(np.ceil(nims / nc))

    # pad array so it matches the product nc * nr
    padding = nc * nr - nims
    heights, widths, num_channels = zip(*[im.shape for im in images])
    h = np.max([im.shape[0] for im in images])
    w = np.max([im.shape[1] for im in images])
    num_channels = np.max([im.shape[2] for im in images])
    ims = [pad(im, new_width=w, new_height=h, new_num_channels=num_channels) for im in images]
    ims += [fill_value * np.ones((h, w, num_channels))] * padding
    coll = np.stack(ims, axis=3)
    coll = np.reshape(coll, (h, w, num_channels, nr, nc))
    # 0  1  2   3   4
    # y, x, ch, co, ro
    if bw:
        # pad each patch by border if requested
        coll = np.append(coll, bv * np.ones((bw,) + coll.shape[1: 5]), axis=0)
        coll = np.append(coll, bv * np.ones((coll.shape[0], bw) + coll.shape[2: 5]), axis=1)
    if transpose:
        nim0 = nc
        nim1 = nr
        if transposeIms:
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
        if transposeIms:
            dim0 = w
            dim1 = h
            #                          nc w  nr h  ch
            coll = np.transpose(coll, (3, 1, 4, 0, 2))
        else:
            dim0 = h
            dim1 = w
            #                          nc h  nr w  ch
            coll = np.transpose(coll, (3, 0, 4, 1, 2))
    coll = np.reshape(coll, ((dim0 + bw) * nim0, (dim1 + bw) * nim1, num_channels))

    return coll


def crop_bounds(images, apply=True, crop_global=True, background=0):
    # pre-compute cropping bounds (tight bounding box around non-background colored pixels)
    if not isinstance(images, list):
        images = [images]
    nzs = [np.where(np.sum(np.atleast_3d(im) != background, axis=2) > 0) for im in images]
    xmins = [np.min(nz[1]) if len(nz[1]) else 0 for nz in nzs]
    xmaxs = [np.max(nz[1]) + 1 if len(nz[1]) else im.shape[1] for nz, im in zip(nzs, images)]  # +1 to allow easier indexing
    ymins = [np.min(nz[0]) if len(nz[0]) else 0 for nz in nzs]
    ymaxs = [np.max(nz[0]) + 1 if len(nz[0]) else im.shape[0] for nz, im in zip(nzs, images)]  # +1 to allow easier indexing
    if crop_global:
        # fix cropping boundaries for all images
        xmins = [np.min(xmins) for _ in xmins]
        xmaxs = [np.max(xmaxs) for _ in xmaxs]
        ymins = [np.min(ymins) for _ in ymins]
        ymaxs = [np.max(ymaxs) for _ in ymaxs]

    if apply:
        images = [np.atleast_3d(images[i])[ymins[i]: ymaxs[i], xmins[i]: xmaxs[i], :] for i in range(len(images))]
    return dict(images=images, xmins=xmins, xmaxs=xmaxs, ymins=ymins, ymaxs=ymaxs)


def split_patches(h, w, patch_size=128, min_patch_size=1, overlap=8, padding='repeat'):
    """given height and width (of a texture) as well as a patch size, return ranges of x and y coordinates dividing the
    texture into patches via slicing"""
    def cl(i, length):
        return np.clip(i, 0, length - 1)

    assert h > min_patch_size, 'height %d is smaller than %d' % (h, min_patch_size)
    assert w > min_patch_size, 'width %d is smaller than %d' % (w, min_patch_size)

    def divide(length):
        starts = np.r_[0:length:patch_size] - overlap
        ends = starts + patch_size + 2 * overlap
        # check that last segment is larger than min_patch_size
        if len(starts) > 1 and length - starts[-1] < min_patch_size:
            # move last starting point back so that last segment is exactly min_patch_size
            ends[-2:] -= min_patch_size - (length - starts[-1])
            starts[-1:] -= min_patch_size - (length - starts[-1])
        if padding == 'repeat':
            return [cl(np.r_[i0:i1], length) for i0, i1 in zip(starts, ends)]
        elif padding is None or padding == 'none':
            return [np.r_[cl(i0, length): cl(i1, length + 1)] for i0, i1 in zip(starts, ends)]
        else:
            raise Exception('padding must be one of: "none", "repeat"')

    xs = divide(w)
    ys = divide(h)
    return xs, ys


def loadmat(filename):
    """wrapper around scipy.io.loadmat that avoids conversion of nested matlab structs to np.arrays"""
    import scipy.io as spio
    mat = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    for key in mat:
        if isinstance(mat[key], spio.matlab.mio5_params.mat_struct):
            mat[key] = to_dict(mat[key])
    return mat


def to_dict(matobj):
    """construct python dictionary from matobject"""
    import scipy.io as spio
    output = {}
    for fn in matobj._fieldnames:
        val = matobj.__dict__[fn]
        if isinstance(val, spio.matlab.mio5_params.mat_struct):
            output[fn] = to_dict(val)
        else:
            output[fn] = val
    return output


def strparse2(strings, pattern, numeric=False, *args):
    res = [re.match(pattern, string) for string in strings]
    matching = np.nonzero(np.array([not r is None for r in res]))[0]
    res = np.array(res)[matching]
    res = np.array([r.groups() for r in res])
    if len(matching) and numeric:
        if len(args) == 1:
            res = res.astype(args[0])
        elif len(args) == res.shape[1]:
            resOut = []
            for ci in range(len(args)):
                resOut.append(res[:, ci].astype(args[ci]))
            res = resOut
        elif len(args) != 0:
            raise Exception('number of type specifiers must equal the number of matching groups in the pattern!')
    return res, matching


def strparse(strings, pattern, numeric=False, *args):
    res, matching = strparse2(strings, pattern, numeric, *args)
    return res


def read_exr(fname, outputType=np.float16):
    import pyexr
    file = pyexr.open(fname)
    channels = file.channel_map['all']
    pixels = file.get(group='all', precision=pyexr.FLOAT)
    return pixels, channels


def read_openexr(fname, channels=None, pixel_type=None, sort_rgb=False):
    """read OpenEXR images"""

    import OpenEXR
    import Imath

    if pixel_type is None or isinstance(pixel_type, str) and pixel_type.lower() == 'float':
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        pixel_type_numpy = np.float32
    elif isinstance(pixel_type, str) and pixel_type.lower() == 'half':
        pixel_type = Imath.PixelType(Imath.PixelType.HALF)
        pixel_type_numpy = np.float16
    elif isinstance(pixel_type, str) and (pixel_type.lower() == 'int' or pixel_type.lower() == 'uint'):
        pixel_type = Imath.PixelType(Imath.PixelType.UINT)
        pixel_type_numpy = np.uint32
    else:
        raise Exception('unsupported pixel type')

    inds = channels

    file = OpenEXR.InputFile(fname)
    header = file.header()
    dw = header['dataWindow']
    sz = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    channels = list(header['channels'].keys())

    if inds is not None:
        if len(inds) == 1 and inds[0] < 0:
            channels = channels[:inds[0]]
        else:
            channels = [channels[ind] for ind in inds]

    if sort_rgb:
        def bgr2rgb(key):
            k = key[-1].lower()
            if k == 'r':
                k = '0'
            elif k == 'g':
                k =  '1'
            elif k == 'b':
                k =  '2'
            return key[:-1] + k
        channels.sort(key=bgr2rgb)

    pixels = file.channels(channels, pixel_type)
    pixels = [np.frombuffer(pix, dtype=pixel_type_numpy).reshape(sz) for pix in pixels]
    pixels = np.stack(pixels, axis=2)

    return pixels, channels


def clamp(arr, lower=0, upper=1):
    if isinstance(arr, np.ndarray):
        arr = arr.clip(lower, upper)
    else:
        if isinstance(arr, torch.Tensor):
            arr = arr.clamp(lower, upper)
        else:
            raise Exception('not implemented for data type ' + str(type(arr)))
    return arr


def write_mp4(frames, fname, extension='jpg', cleanup=True, fps=25, crf=10, scale=1, gamma=1,
              ffmpeg='/usr/bin/ffmpeg', digit_format='%04d', quality=95, verbosity=1):
    """Write a sequence of frames as mp4 video file by writing temporary images and converting them with ffmpeg.

    Arguments:
    frames -- list / array of 2D or 3D numpy arrays holding grayscale or RGB images, alternatively this can be a list
              of filenames to images already written to disk; the expected filename format is prefix_%04d.extension
    fname -- relative / absolute path to output mp4 file (including extension)
    
    Keyword arguments:
    extension -- file extension of temporary images (default 'jpg')
    cleanup -- set this to True to remove temporary images (default True)
    fps -- frames per second of output video (default 25)
    crf -- constant rate factor parameter to ffmpeg, scalar between 0 (lossless) and 51 (worst quality) (default 10)
    scale -- tonemapping scale, used to brighten or darken images (default 1.0)
    gamma -- tonemapping gamma, used to stretch / squeeze dark or bright areas (default 1.0)
    ffmpeg -- path to ffmpeg executable (default /usr/bin/ffmpeg)
    digit_format -- numerical format of the running index in the filenames
    """
    import os
    from PIL import Image
    import tempfile
    import subprocess
    tmp = tempfile.TemporaryDirectory().name
    os.makedirs(tmp)

    if isinstance(frames[0], np.ndarray):
        for fi in range(len(frames)):
            frame = np.atleast_3d(frames[fi])
            if frame.shape[0] % 2 != 0:
                frame = frame[1:, :, :]
            if frame.shape[1] % 2 != 0:
                frame = frame[:, 1:, :]
            if frame.ndim == 3 and frame.shape[2] == 1:
                frame = frame[:, :, 0]
            im = Image.fromarray((255 * np.clip(scale * frame, 0., 1.) ** (1. / gamma)).astype(np.uint8))
            if verbosity > 1:
                print('writing image to ' + os.path.join(tmp, 'frame_%04d.%s' % (fi, extension)))
            kwargs = dict()
            if extension.lower() == 'jpg':
                kwargs['quality'] = quality
            im.save(os.path.join(tmp, 'frame_' + (digit_format % fi) + '.' + extension), **kwargs)
        prefix = os.path.join(tmp, 'frame_')
    else:
        if not isinstance(frames[0], str):
            raise Exception('frames should be list of np.ndarrays or filenames')
        prefix = frames[0]
        prefix = prefix[:prefix.rfind('_') + 1]

    cmd = [ffmpeg, '-y', '-framerate', str(fps), '-i', prefix + digit_format + '.' + extension, '-c:v',
           'libx264', '-vf', 'fps=%d' % fps, '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-preset', 'veryslow', '-pix_fmt', 'yuv420p', '-crf', str(crf), fname]
    if verbosity > 0:
        print(' '.join(cmd))
    stdout = subprocess.STDOUT if verbosity > 1 else subprocess.DEVNULL
    res = subprocess.run(cmd, stdout=stdout, stderr=subprocess.STDOUT)

    if cleanup:
        for fi in range(len(frames)):
            os.remove(prefix + (digit_format % fi) + '.' + extension)

    return prefix


def blur_image(image, blur_size=49, use_torch=False, filter_type='gauss'):
    if blur_size <= 1:
        return image.copy()
    image = np.atleast_3d(image.astype(np.float32))
    if use_torch:
        import torch
        from torch.nn.functional import conv2d
        def convolve(image, kernel):
            dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            kernel = kernel.astype(image.dtype)
            with torch.no_grad():
                h, w = kernel.shape[:2]
                if h % 2 == 0:
                    p = (h // 2 - 1, h - h // 2, w // 2 - 1, w - w // 2)
                else:
                    p = (h // 2, h - h // 2 - 1, w // 2, w - w // 2 - 1)
                padding = torch.nn.ReflectionPad2d(p)
                image = torch.tensor(image.transpose(2, 0, 1)[:, None])
                image = padding(image)
                result = conv2d(image.to(dev), torch.tensor(kernel.transpose(2, 0, 1))[None].to(dev))
                result = result.cpu().numpy()[:, 0, :, :].transpose((1, 2, 0))
            return result
    else:
        from scipy.ndimage import convolve

    if filter_type == 'pyramid':
        N = blur_size // 2
        n = np.linspace(0, 1, N)
        g = np.r_[n[:-1], n[::-1]]
        kernel = g[:, None] * g[None, :]
    elif filter_type == 'triangle':
        N = blur_size // 2
        ys, xs = np.mgrid[-N : N + 1, -N : N + 1]
        kernel = np.maximum(np.abs(ys), np.abs(xs))
        kernel = N - kernel
    elif filter_type == 'gauss':
        N = blur_size - 1
        n = np.mgrid[0: N + 1] - N / 2
        g = np.exp(-0.5 * (2.5 * n / (N / 2)) ** 2)
        kernel = g[:, None] * g[None, :]
    kernel = np.atleast_3d(kernel / np.sum(kernel))
    return convolve(image, kernel)


def qimage_to_np(im):
    ptr = im.bits()
    ptr.setsize(im.byteCount())
    arr = np.array(ptr).reshape((im.height(), im.width(), -1))
    return arr


def sortrows(arr, order=None):
    if order is None:
        return arr[np.lexsort(arr.T[::-1])]
    else:
        keys = arr.T[order]
        return arr[np.lexsort(keys[::-1])]


class Dct(Dict):
    """Dictionary with nicer formatting and dot notation access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exp:
            raise AttributeError('Key ' + str(key) + ' does not exist') from exp

    def __setattr__(self, key, val):
        self[key] = val

    def __repr__(self):
        if not len(self):
            return ""
        width = max([len(str(k)) for k in self])
        items = '{:' + str(width + 2) + 's} {}'
        items = [items.format(str(key) + ':', self[key]) for key in sorted(self.keys())]
        items = '\n'.join(items)
        return items

