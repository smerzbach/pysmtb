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
