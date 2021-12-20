"""
Created on Wed Jan 14 23:01:57 2020

@author: Sebastian Merzbach <smerzbach@gmail.com>
"""

import numpy as np
import os
import re
import subprocess
import sys
from typing import Dict, Union

try:
    import torch
except:
    pass

# backwards compatibility after refactoring
from pysmtb.image import assign_masked, annotate_image, pad, collage, crop_bounds, split_patches, \
    read_exr, read_openexr, write_openexr, tonemap, blur_image, qimage_to_np


def execute(args: Union[str, list],
          logfile: str = None,
          universal_newlines: bool = True,
          shell: bool = False,
          **kwargs):
    """run external command (by default with low priority), capture and print its output; returns return code and log"""
    if logfile is not None:
        logfile = open(logfile, 'w')

    creationflags = 0
    try:
        # only available on Windows...
        creationflags |= subprocess.IDLE_PRIORITY_CLASS
    except:
        pass

    log = []
    process = subprocess.Popen(args,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               creationflags=creationflags,
                               universal_newlines=universal_newlines,
                               shell=shell,
                               **kwargs)

    while True:
        # print command output and write to log if requested
        output = process.stdout.readline()
        returncode = process.poll()
        if output == '' and returncode is not None:
            # process finished
            break
        if output:
            print(output.strip())
            log.append(output)
            if logfile is not None:
                logfile.write(output)

    if logfile is not None:
        logfile.close()
    return returncode, log


def find_dim(inp, size=3):
    """return first dimension that matches the given size"""
    dim = np.where(np.array(inp.shape) == size)[0]
    if not len(dim):
        raise Exception('none of the input dimensions is %d: %s' % (size, str(inp.shape)))
    return dim[0]


def dims_execpt(inp, dim):
    """return all except the specified dimension indices"""
    return tuple(np.r_[[d for d in range(inp.ndim) if d != dim]])


def sizes_execpt(inp, dim):
    """return all except the specified dimension indices"""
    return [s for d, s in enumerate(inp.shape) if d != dim]


def replace_dim(inp, dim: int, new_size: int):
    """given input array or tensor, return shape of array with one of the dimensions (dim, can be negative) replaced
     by the specified new size"""
    nd = len(inp)
    return [new_size if d == dim or dim < 0 and d == nd + dim else s for d, s in enumerate(inp)]


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


def clamp(arr, lower=0, upper=1):
    if isinstance(arr, np.ndarray):
        arr = arr.clip(lower, upper)
    else:
        if isinstance(arr, torch.Tensor):
            arr = arr.clamp(lower, upper)
        else:
            raise Exception('not implemented for data type ' + str(type(arr)))
    return arr


def video_writer(filename: str, vcodec: str = 'libx264', framerate: float = 25,
                 lossless: bool = False, quality: float = 0.75, pix_fmt: str = 'yuv420p',
                 loop: Union[bool, int] = 0, verbosity: int = 0, ffmpeg_path=None, **kwargs):
    import skvideo

    # override ffmpeg to support more codecs (e.g. webp is not supported by conda's ffmpeg)
    if ffmpeg_path is None:
        ffmpegs = []
        for path in os.get_exec_path():
            ffmpeg = os.path.join(path, 'ffmpeg.exe' if sys.platform == 'win32' else 'ffmpeg')
            if os.path.exists(ffmpeg) and os.access(path, os.X_OK):
                ffmpegs.append(ffmpeg)
        if '/usr/bin/ffmpeg' in ffmpegs:
            # prefer system ffmpeg over any bundled version
            ffmpeg_path = '/usr/bin/ffmpeg'
    if ffmpeg_path is not None:
        if not os.path.isdir(ffmpeg_path):
            # make sure we strip ffmpeg(.exe) from the provided path, as skvideo.setFFmpegPath() expects the containing
            # directory only
            path, exec = os.path.split(ffmpeg_path)
            if exec.startswith('ffmpeg'):
                ffmpeg_path = path
        skvideo.setFFmpegPath(ffmpeg_path)

    # the order of this import is relevant (needs to come after ffmpeg path was set)!
    import skvideo.io

    if isinstance(loop, bool):
        # -loop 0 means endless looping
        loop = 0 if loop else (-1 if vcodec == 'gif' else 1)

    indict = {'-framerate': str(framerate)}
    outdict = {
        '-vcodec': vcodec,
        '-framerate': str(framerate),
    }

    if not (0 <= quality and quality <= 1):
        raise Exception('quality must be in [0, 1]')

    if vcodec in ['libx264']:
        profile = kwargs.pop('profile', 'high')

        outdict.update({
            '-profile:v': profile,
            '-level:v': '4.0',
            '-pix_fmt': pix_fmt,
            '-filter_complex': '[0]pad=ceil(iw/2)*2:ceil(ih/2)*2',
        })

        preset = kwargs.pop('preset', 'high')
        if preset not in ['lowest', 'lower', 'low', 'high', 'higher', 'highest']:
            raise ValueError('for x264, preset must be one of lowest, lower, low, high, higher, highest')

        crf = int(1 + 62 * (1 - quality))  # crf goes from 0 to 63, 0 being best quality
        # crf = int(63 * (1 - quality))  # crf goes from 0 to 63, 0 being best quality
        outdict['-q:v'] = str(int(quality * 100))
        outdict['-crf'] = str(crf)

    elif vcodec == 'libwebp':
        # setting libwebp explicitly fails, so let's rely on ffmpeg's auto detection
        outdict.pop('-vcodec', None)

        preset = kwargs.get('preset', 'default')
        if preset not in ['none', 'default', 'picture', 'photo', 'drawing', 'icon', 'text']:
            raise ValueError('for webp, preset must be one of none, default, picture, photo, drawing, icon, text')

        outdict['-preset'] = str(preset)
        outdict['-loop'] = str(loop)
        outdict['-compression_level'] = str(kwargs.pop('compression_level', 4))  # 0-6
        if quality >= 1 or lossless:
            outdict['-lossless'] = '1'
        else:
            outdict['-q:v'] = str(int(quality * 100))

    elif vcodec == 'gif':
        outdict['-loop'] = str(loop)
        outdict['-final_delay'] = str(kwargs.pop('final_delay', '-1'))  # centi seconds
    else:
        raise NotImplementedError('video codec %s not implemented' % vcodec)
    for key, value in kwargs.items():
        if not key.startswith('-'):
            key = '-' + key
        outdict[key] = str(value)
    writer = skvideo.io.FFmpegWriter(filename, inputdict=indict, outputdict=outdict, verbosity=verbosity)
    return writer


def write_video(filename: str, frames: Union[np.ndarray, list], offset: float = 0.0, scale: float = 1.0, gamma: float = 1.0,
                masks: Union[np.ndarray, list, bool] = None, background: Union[np.ndarray, list, tuple] = None,
                loop: bool = False,
                ffmpeg_path=None, verbosity: int = 0, **kwargs):
    """given a sequence of frames (as 4D np.ndarray or as list of 3D np.ndarrays), export a video using FFMPEG;

    codecs:
    the codec is automatically derived from the file extension, currently supported are:
     mp4, webp and gif

    webp: it might be necessary to specify the path to the system ffmpeg since the one bundled with conda lacks webp
    support

    tonemapping:
    the offset, scale & gamma arguments can be used to apply basic tonemapping

    transparency / alpha blending:
    it is possible to export transparent videos via webp (no support with other codecs) by providing alpha masks;
    alternatively, the video sequence can be alpha blended against a background, either a constant value, color, or a
    static background image

    loop sequence
    set loop = True for poor man's looped animations where the reversed animation is appended (doubling the number of
    frames as a side effect)

    additional arguments:
    all additional arguments will by passed through to the ffmpeg video writer, examples are:
    vcodec: str = 'libx264'
    framerate: float = 25
    lossless: bool = False (only supported for webp)
    quality: float = 0.75 (\in [0, 1])
    profile: str (for x64 (mp4): \in {'lowest', 'lower', 'low', 'high', 'higher', 'highest'},
             for webp: \in {'none', 'default', 'picture', 'photo', 'drawing', 'icon', 'text'})
    pix_fmt: str = 'yuv420p'
    loop: Union[bool, int] = 0
    verbosity: int = 0
    ffmpeg_path=None
    """

    requires_tonemapping = offset != 0. or scale != 1. or gamma != 1.
    if frames[0].dtype == np.uint8:
        if requires_tonemapping:
            raise Exception('frames are already in uint8 format but tonemapping is requested')
        if masks is not None and masks[0].dtype != np.uint8:
            raise Exception('frames are in uint8 format but masks are not')
        if background is not None:
            raise Exception('frames are already in uint8 format but alpha blending is requested (background != None)')
    else:
        # non uint8 needs to be scaled and converted
        requires_tonemapping = True

    kwargs.update(dict(ffmpeg_path=ffmpeg_path))
    ext = os.path.splitext(filename)[1].lower()
    if ext in ['.mp4', '.avi']:
        writer = video_writer(filename=filename, verbosity=verbosity, **kwargs)
    elif ext == '.webp':
        # writer = video_writer(filename=filename, verbosity=verbosity, vcodec='libwebp', **kwargs)
        writer = video_writer(filename=filename, verbosity=verbosity, vcodec='libwebp', **kwargs)
    elif ext == '.gif':
        writer = video_writer(filename=filename, verbosity=verbosity, vcodec='gif', **kwargs)
    else:
        raise NotImplementedError('unexpected file extension: ' + ext)

    frame_inds = np.r_[:len(frames)]
    if loop:
        frame_inds = np.r_[frame_inds, frame_inds[::-1]]

    for fi in frame_inds:
        frame = frames[fi]
        if masks is not None:
            mask = masks[fi]
        else:
            mask = None
        if requires_tonemapping:
            frame = tonemap(frame, offset=offset, scale=scale, gamma=gamma, as_uint8=True, alpha=mask, background=background)
        elif mask is not None:
            # no tonemapping, no alpha blending, just concatenate alpha channel for transparency
            frame = np.concatenate((np.atleast_3d(frame), np.atleast_3d(mask)), axis=2)
        writer.writeFrame(frame)
    writer.close()


def write_mp4(frames, fname, extension='jpg', cleanup=True, fps=25, crf=10, scale=1, gamma=1,
              ffmpeg='/usr/bin/ffmpeg', digit_format='%04d', quality=95, verbosity=1):
    from warnings import warn
    warn('write_mp4() is deprecated, use write_video() instead')
    write_video(filename=fname, frames=frames, fps=fps, quality=1. - crf / 63, scale=scale, gamma=gamma,
                ffmpeg_path=ffmpeg)


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
