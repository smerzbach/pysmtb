from copy import deepcopy
try:
    import colour
except ModuleNotFoundError:
    colour = None
    pass
import numpy as np
try:
    import torch
except:
    torch = None

from pysmtb import iv
from pysmtb.utils import collage


def rand_im(width, height, num_channels, num_ims, scales):
    import opensimplex
    ims = np.zeros((width, height, num_channels, num_ims))
    tmp = opensimplex.OpenSimplex()
    for x in range(width):
        for y in range(height):
            for c in range(num_channels):
                for i in range(num_ims):
                    ims[x, y, c, i] = tmp.noise4d(x / scales[0], y / scales[1], c / scales[2], i)
    return ims


if __name__ == "__main__":
    # test colormapping
    v = iv.iv(np.concatenate((np.einsum('ic,jc->ijc', np.random.rand(200, 1), np.random.rand(500, 1)),
                              np.linspace(0, 1, 500)[None, :, None].repeat(200, axis=0)), axis=0))

    # test zoomed copy
    from time import sleep
    np.random.seed(1)
    v = iv.iv(np.einsum('ic,jc->ijc', np.random.rand(4,3), np.random.rand(5, 3)))
    import matplotlib.pyplot as plt
    v.copy_to_clipboard_zoomed()
    v.copy_to_clipboard_zoomed()
    v.save('~/export_canvas.png', canvas=True)

    # test boolean images
    im1 = np.random.rand(10, 10, 3) < 0.5 ** 3
    v = iv.iv(im1, np.zeros((100, 100), dtype=bool), np.ones((100, 100), dtype=bool))

    # test cropping
    im1 = np.zeros((100, 100), dtype=bool)
    im1[25:-25, 25:-25] = True
    v = iv.iv(im1, np.zeros((10, 10), dtype=bool))

    # test tight collage mode
    ims1 = [np.random.rand(25, 15, 3) for _ in range(10)]
    ims2 = [np.random.rand(10, 12, 3) for _ in range(10)]
    ims3 = [np.random.rand(15, 12, 1) for _ in range(8)]
    coll = collage(ims1 + ims2 + ims3, bw=1, tight=False, nc=5)
    coll_tight = collage(ims1 + ims2 + ims3, bw=1, tight=True, nc=5)
    v = iv.iv(dict(tight=coll_tight, non_tight=coll), collage=True, collageBorderWidth=1, collageBorderValue=1, annotate=True)
    v = iv.iv(ims1, ims2, ims3, collage=True, collageBorderWidth=1)

    if color is not None:
        # def interval(n):
        #     return deepcopy(cmfs).align(colour.SpectralShape(250., 800., (800. - 250.) / n)).shape.interval
        #
        # intervals = {n: interval(n) for n in range(6, 100)}

        wl0 = 250.
        wl1 = 800.
        wl_range = wl1 - wl0
        num_bands = 30
        spec_shape = colour.SpectralShape(wl0, wl1, wl_range / (num_bands - 1))

        cmfs = deepcopy(colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer'])
        cmfs = cmfs.align(shape=spec_shape)

        illuminant = colour.SDS_ILLUMINANTS['D65']
        illuminant = illuminant.align(spec_shape)
        im = colour.msds_to_XYZ(np.random.rand(10, 10, num_bands), cmfs, illuminant, method='Integration', shape=spec_shape)
        iv.iv(im)

    # test mixed inputs & collage
    if torch is not None:
        dims = np.zeros((4, 5, 2))
        dims[:2, :, 0] = 10
        dims[2:, :, 0] = 15
        c = collage([np.random.rand()])
        ims = np.random.rand(10, 10, 3, 4)
        # ims = rand_im(100, 100, 3, 4, [10, 10, 3])
        ims_tensors = torch.rand((12, 3, 15, 15))
        ims_list = [np.random.rand(12, 12) for _ in range(9)]
        v = iv.iv(ims, ims_tensors, ims_list, collage=True, collageBorderWidth=2)

    # compare different illuminants / CMFs in GUI
    ims_spec = np.random.rand(10, 10, 31, 4).astype(np.float32)
    v = iv.iv(ims_spec, autoscale=False)

    # check display of labels
    iv.iv(dict(first=np.random.rand(100, 100, 3) + np.linspace(-1, 1, 100)[None, :, None],
               second=np.random.rand(100, 100, 3) + np.linspace(-1, 1, 100)[:, None, None]),
          annotate=True, annotate_numbers=False)

    # WIP: overlay of numeric pixel values
    ims = np.random.rand(10, 10, 3, 16).astype(np.float16)
    v = iv.iv(ims)
    v.overlay_pixel_values()

    print()
