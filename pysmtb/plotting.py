from typing import List, Tuple, Union
import matplotlib
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('Qt5Agg')


def plot3(xyz, *args, **kwargs):
    return _plot3d('plot', xyz, *args, **kwargs)


def scatter3(xyz, *args, **kwargs):
    return _plot3d('scatter', xyz, *args, **kwargs)


def quiver3(xyz, uvw, *args, **kwargs):
    return _plot3d('quiver', xyz, uvw, *args, **kwargs)


def text3(xyz, strings=None, *args, **kwargs):
    if strings is None:
        n = xyz.size // 3
        strings = ['%d' % i for i in range(n)]
    return _plot3d('text', xyz, strings, *args, **kwargs)


def _plot3d(method, xyz, second=None, axes=None, axis='equal', limits=None, clip=False, *args, **kwargs):
    if axes is None:
        fig = plt.gcf()
        axes = fig.add_subplot(111, projection='3d')
        axes_provided = False
    else:
        axes_provided = True

    xyz = np.atleast_2d(xyz)
    if xyz.ndim == 2 and xyz.shape[1] == 3 and xyz.shape[0] != 3:
        xyz = xyz.T
    x = xyz[0, :]
    y = xyz[1, :]
    z = xyz[2, :]

    if method in ['quiver', 'text'] and second is None:
        raise Exception('method %s requires "second" argument to be set' % method)

    if limits is not None and len(limits):
        xlim = limits[:2]
        ylim = limits[2:4]
        zlim = limits[4:6]
    elif axis == 'equal':
        max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5
        xlim = [mid_x - max_range, mid_x + max_range]
        ylim = [mid_y - max_range, mid_y + max_range]
        zlim = [mid_z - max_range, mid_z + max_range]
    elif axis == 'fixed':
        xlim = None
        ylim = None
        zlim = None
    elif axis != 'auto':
        raise NotImplementedError('axis mode %s not implemented' % axis)
    else:
        xlim = [x.min(), x.max()]
        ylim = [y.min(), y.max()]
        zlim = [z.min(), z.max()]

    if clip:
        x = x.copy()
        y = y.copy()
        z = z.copy()
        mask = (x < xlim[0]) | (x > xlim[1]) | (y < ylim[0]) | (y > ylim[1]) | (z < zlim[0]) | (z > zlim[1])
        x[mask] = np.nan
        y[mask] = np.nan
        z[mask] = np.nan

    if method == 'scatter':
        handle = axes.scatter(x, y, z, *args, **kwargs)
    elif method == 'plot':
        handle = axes.plot(x, y, z, *args, **kwargs)
    elif method == 'quiver':
        if second.ndim == 2 and second.shape[1] == 3 and second.shape[0] != 3:
            second = second.T
        u = second[0, :]
        v = second[1, :]
        w = second[2, :]
        handle = axes.quiver(x, y, z, u, v, w, *args, **kwargs)
    elif method == 'text':
        strings = second
        handle = []
        for xi, yi, zi, si in zip(x, y, z, strings):
            handle.append(axes.text(xi, yi, zi, si, *args, **kwargs))
    else:
        raise NotImplementedError('method %s has not been implemented')

    if axes_provided:
        # expand previous axes limits
        xlim[0] = np.minimum(xlim[0], axes.get_xlim()[0])
        xlim[1] = np.maximum(xlim[1], axes.get_xlim()[1])
        ylim[0] = np.minimum(ylim[0], axes.get_ylim()[0])
        ylim[1] = np.maximum(ylim[1], axes.get_ylim()[1])
        zlim[0] = np.minimum(zlim[0], axes.get_zlim()[0])
        zlim[1] = np.maximum(zlim[1], axes.get_zlim()[1])

    if axis != 'fixed':
        axes.set_xlim(xlim[0], xlim[1])
        axes.set_ylim(ylim[0], ylim[1])
        axes.set_zlim(zlim[0], zlim[1])

    axes.set_position(Bbox([[0, 0], [1, 1]]))
    return handle


def annotate_spec_images(images: Union[np.ndarray, List],
                         wavelengths: Union[np.ndarray, List],
                         annotate_pixels: Union[List, Tuple] = (),
                         scale: float = 1.0,
                         offset: float = 0.0,
                         gamma: float = 1.0,
                         illuminant_name: str = 'E',
                         cmf_name: str = 'CIE 1931 2 Degree Standard Observer',
                         interpolation: str = 'nearest',
                         annot_color: tuple = (1, 0, 1),
                         annot_x_offset: float = 5.,
                         annot_y_offset: float = 5.,
                         ) -> Tuple:
    """display subplots of multiple spectral images, each annoated with line plots of individual pixels' spectra"""
    from pysmtb.image import spec_image_to_srgb, tonemap
    if isinstance(images, np.ndarray):
        images = [images[..., i] for i in range(images.shape[3])]
    images_rgb = []
    for image in images:
        images_rgb.append(spec_image_to_srgb(image, wavelengths=wavelengths, illuminant_name=illuminant_name, cmf_name=cmf_name))
        images_rgb[-1] = tonemap(images_rgb[-1], offset=offset, scale=scale, gamma=gamma, as_uint8=True)

    n = len(images)
    nc = int(np.ceil(np.sqrt(n)))
    nr = int(np.ceil(n / nc))

    y_min = np.inf
    y_max = -np.inf
    fig, axes = plt.subplots(nrows=nr, ncols=nc * 2)
    axes_spec = axes[:, 1::2]
    axes = axes[:, 0::2]
    for i, (ax_im, ax_spec) in enumerate(zip(axes.flatten(), axes_spec.flatten())):
        if i < len(images):
            ax_im.imshow(images_rgb[i], interpolation=interpolation)
            for j, pix in enumerate(annotate_pixels):
                ax_spec.plot(wavelengths, images[i][pix[0], pix[1], :], color=images_rgb[i][pix[0], pix[1]].astype(np.float32) / 255, label=str(j))
                ylim = ax_spec.get_ylim()
                y_min = np.minimum(y_min, ylim[0])
                y_max = np.maximum(y_max, ylim[1])
                ax_im.scatter(pix[1], pix[0], 50, marker='o', edgecolor=[1, 1, 1], facecolor='none')
                ax_im.text(pix[1] + annot_x_offset, pix[0] + annot_y_offset, str(j), color=annot_color)
            ax_spec.legend()
        else:
            ax_im.set_visible(False)
            ax_spec.set_visible(False)

    for ax in axes_spec.flatten():
        ax.set_ylim((y_min, y_max))
    return fig, axes, axes_spec


def spec_imshow(image: np.ndarray,
                wavelengths: Union[np.ndarray, List],
                annotate_pixels: Union[List, Tuple] = (),
                scale: float = 1.0,
                offset: float = 0.0,
                gamma: float = 1.0,
                illuminant_name: str = 'E',
                cmf_name: str = 'CIE 1931 2 Degree Standard Observer',
                interpolation: str = 'nearest') -> None:
    """display tonemapped spectral image, optionally surrounded by (up to 8) spectra sampled from its pixels"""
    from pysmtb.image import spec_image_to_srgb, tonemap
    image_rgb = spec_image_to_srgb(image, wavelengths=wavelengths, illuminant_name=illuminant_name, cmf_name=cmf_name)
    image_rgb = tonemap(image_rgb, offset=offset, scale=scale, gamma=gamma, as_uint8=True)

    nrows = 4
    ncols = 6
    n_ax_im = 4 * 4

    ax = plt.subplot2grid((4, 6), (0, 1), rowspan=4, colspan=4)
    ih = ax.imshow(image_rgb, aspect='equal', interpolation=interpolation)

    n = len(annotate_pixels)
    assert n <= 8, 'select at most 8 spectra'
    spectra = []
    for i, (x, y) in enumerate(annotate_pixels):
        spectra.append((image[y, x, :], image_rgb[y, x, :]))
        ax.scatter(x, y, s=250, marker='o', facecolors='none', edgecolors='r', linewidth=2)
        ax.text(x + 2, y + 2, '%d' % (i + 1), color='r')

    axes = []
    for i in range(n):
        index = i if i < 4 else i + n_ax_im
        index = index // nrows + (index % nrows) * ncols
        axes.append(plt.subplot(nrows, ncols, index + 1))

    for i, spec in enumerate(spectra):
        axes[i].plot(wavelengths, spec[0], color=np.clip(spec[1].astype(np.float32) / 255, 0, 1))
        axes[i].set_yticklabels([])
        x0, x1 = axes[i].get_xlim()
        y0, y1 = axes[i].get_ylim()
        dx = (x1 - x0) / 10
        dy = (y1 - y0) / 8
        axes[i].text(x0 + dx, y1 - dy, '%d' % (i + 1), color='r')
