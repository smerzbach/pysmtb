from copy import deepcopy
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Union

def assign_masked(mask: np.ndarray, values: np.ndarray, buffer: np.ndarray = None, init_val: float = 0.):
    """given an h x w binary mask, an array with nnz(mask) x nc_vals elements, and optionally a buffer (either empty or
    pre-allocated with h * w * nc_vals elements), assign the values array at those entries of buffer where mask is
    True; return buffer with shape h x w x nc_vals"""
    s = np.atleast_2d(values).shape
    num_vals = s[0]
    nc_vals = s[1:]
    h, w, _ = np.atleast_3d(mask).shape
    nnz_mask = np.count_nonzero(mask)
    assert nnz_mask == num_vals, 'values must be nnz_mask x nc'
    if buffer is None:
        buffer = init_val * np.ones((h, w) + nc_vals, dtype=np.float32)
    assert buffer.size == h * w * np.prod(nc_vals), 'buffer must have %d * %d * %d elements, got %d' % (h, w, nc_vals, buffer.size)
    buffer = buffer.reshape((h * w,) + nc_vals)
    buffer[mask.ravel(), :] = values
    buffer = buffer.reshape((h, w) + nc_vals)
    return buffer


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


def collage(images, dim=-1,
            nc=None, # number of columns
            nr=None, # number of rows
            bw=0, # border width
            bv=0, # border value
            transpose=False,
            transpose_ims=False,
            tight=True, # pack images tightly in each row / column (False: all images will be padded to the same size)
            crop=False, # crop margins in each image, enabling even tighter packing
            crop_value=0, # pixel value that will be considered as margin
            crop_global=False, # determine margin globally, i.e., the minimum over all images
            **kwargs):
    """assemble a list of images or an ndarray into an nr x nc mosaic of sub-images, optionally with border separating
    the images, by default sub-images are packed tightly, i.e. they are padded per row and column with the minimal
    necessary margin to allow concatenation; optionally the entire collage or individual images can be transposed"""
    if isinstance(images, np.ndarray):
        # slice into specified dimension of ndarray
        images = np.atleast_3d(images)
        if images.ndim == 3:
            images = images[..., None]
        images = [images.take(i, axis=dim) for i in range(images.shape[dim])]
    if isinstance(images, list):
        images = [np.atleast_3d(im) for im in images]

    if crop:
        images = crop_bounds(images, apply=True, crop_global=crop_global, background=crop_value)['images']

    nims = len(images)

    if 'fill_value' in kwargs:
        from warnings import warn
        warn('"fill_value" is deprecated, use "bv" instead')

    if nc is None:
        if nr is None:
            nc = int(np.ceil(np.sqrt(nims)))
        else:
            nc = int(np.ceil(nims / nr))
    if nr is None:
        nr = int(np.ceil(nims / nc))
    assert nr * nc >= nims, 'specified nr & nc (nr=%d, nc=%d) are too small for %d images' % (nr, nc, nims)

    if nr * nc < nims:
        nc = int(np.ceil(np.sqrt(nims)))
        nr = int(np.ceil(nims / nc))

    # all images have to be padded in channel dimension to the maximum number of channels
    num_channels = np.max([im.shape[2] for im in images])

    if transpose_ims:
        # transpose individual images
        images = [np.transpose(im, (1, 0, 2)) for im in images]

    # fill up array so it matches the product nc * nr; ensure we always fill up at least one array to avoid problems
    # with implicit conversions during the surrounding np.array() call
    ims = np.array(images + [np.empty((0, 0, 0)) for _ in range(nc * nr - nims + 1)], dtype=object)
    ims = ims[:-1]

    if transpose:
        # swap so that nr & nc remain intuitive when transposed
        nr, nc = (nc, nr)

    # arrange into grid
    ims = np.reshape(ims, (nr, nc))

    if transpose:
        # optionally transpose
        ims = ims.T

    # query nr & nc again in case transpose == True
    nr, nc = ims.shape

    # get height & width of each image, arranged as nr x nc array so we can work row- & column-wise
    if ims.size == 1 and np.isscalar(ims.flatten()[0]):
        ims = np.array([[np.atleast_3d(ims[0][0])]], dtype=object)
        heights = [[1]]
        widths = [[1]]
    else:
        heights = np.reshape([im.shape[0] for im in ims.flatten()], ims.shape)
        widths = np.reshape([im.shape[1] for im in ims.flatten()], ims.shape)

    if tight:
        row_heights = np.max(heights, axis=1)
        col_widths = np.max(widths, axis=0)
    else:
        row_heights = np.repeat(np.max(heights), nr)
        col_widths = np.repeat(np.max(widths), nc)

    rows = []
    ii = 0
    for ri in range(nr):
        h = row_heights[ri]
        row = []
        for ci in range(nc):
            w = col_widths[ci]
            if ii < nims:
                im = ims[ri, ci]
                if im.shape[2] == 1:
                    # replicate single-channel images (other channel counts will be zero-padded, e.g. for RG-coded images)
                    im = np.repeat(im, num_channels, axis=2)
                im = pad(im, new_width=w + bw, new_height=h + bw, new_num_channels=num_channels, value=bv)
            else:
                im = bv * np.ones((h + bw, w + bw, num_channels))
            row.append(im)
            ii += 1
        rows.append(np.concatenate(row, axis=1))
    coll = np.concatenate(rows, axis=0)
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


def read_exr(fname, outputType=np.float16):
    import pyexr
    file = pyexr.open(fname)
    channels = file.channel_map['all']
    pixels = file.get(group='all', precision=pyexr.FLOAT)
    return pixels, channels


def read_openexr(fname, channels=None, pixel_type=None, sort_rgb=False) -> Tuple[np.ndarray, list]:
    """read OpenEXR images, returns np.ndarray and list of channel names"""

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


def write_openexr(filename: str, image: np.ndarray, channels: list = None, pixel_types: Union[str, list] = None,
                  compression='PIZ'):
    """write image (stored as 2D/3D numpy.ndarray) in OpenEXR HDR format with channel names specified as list of
    strings; the pixel type (one of float, half or uint) can be specified globally or per channel, if omitted, it is
    chosen according to the np.ndarray's dtype, if this matches any of the three options; compression method (all
    lossless) can be chosen as one of [None, 'NO', 'RLE', 'ZIPS', 'ZIP', 'PIZ', 'PXR24', 'B44', 'B44A', 'DWAA', 'DWAB'],
    default: PIZ

    channel names are automatically assigned for the following cases, unless explicitly specified:
    1 channel: L
    3 channels: RGB
    4 channels: RGBA
    """
    import Imath
    import OpenEXR

    compressions = [k.replace('_COMPRESSION', '') for k in vars(Imath.Compression).keys() if k.endswith('_COMPRESSION')]
    if compression is None:
        compression = 'NO'
    if compression.upper() in compressions:
        compression = Imath.Compression(compression.upper() + '_COMPRESSION')
    else:
        raise Exception('unknown compression format: ' % compression)

    image = np.atleast_3d(image)
    height, width, num_channels = image.shape

    def get_pixel_types(pixel_type: str):
        pixel_type = pixel_type.lower()
        if pixel_type in ['float', 'float32']:
            return Imath.PixelType('FLOAT')
        elif pixel_type in ['half','float16']:
            return Imath.PixelType('HALF')
        elif pixel_type in [bool, 'int', 'uint', 'uint32']:
            return Imath.PixelType('UINT')
        else:
            raise Exception('pixel type %s not supported, must be "float", "half" or "int"' % pixel_type)

    def pixel_type_np(pixel_type: Imath.PixelType):
        if pixel_type == Imath.PixelType('HALF'):
            return np.float16
        elif pixel_type == Imath.PixelType('FLOAT'):
            return np.float32
        elif pixel_type == Imath.PixelType('UINT'):
            return np.uint32
        else:
            raise Exception('unexpected value for pixel type')

    if pixel_types is None:
        if image.dtype in [np.float32, np.float64]:
            pixel_types = 'float'
        elif image.dtype == np.float16:
            pixel_types = 'half'
        elif image.dtype in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32]:
            pixel_types = 'uint'
        else:
            raise Exception('pixel type cannot be derived automatically for array of type ' + str(image.dtype))

    if isinstance(pixel_types, str):
        pixel_types = [pixel_types] * num_channels
    pixel_types = [get_pixel_types(pixel_type) for pixel_type in pixel_types]

    if channels is None:
        # auto assign channel names if none were specified
        if num_channels == 1:
            channels = ['L']
        elif num_channels == 3:
            channels = ['R', 'G', 'B']
        elif num_channels == 3:
            channels = ['R', 'G', 'B', 'A']
        else:
            num_digits = max(1, int(np.ceil(np.log10(num_channels + 1))))
            channels = [('ch%0' + str(num_digits) + 'd') % ci for ci in nrange(num_channels)]
    assert len(channels) == num_channels, 'number of channels (%d) must match the image ' \
                                          'dimensions (%d)' % (len(channels), num_channels)
    assert len(pixel_types) == num_channels, 'pixel type was speficied %d times, should match the image ' \
                                             'dimensions (%d) or be defined once' % (len(pixel_types), num_channels)
    header = OpenEXR.Header(width, height)
    header['compression'] = compression
    header['channels'] = {channel: Imath.Channel(pixel_type) for channel, pixel_type in zip(channels, pixel_types)}
    outfile = OpenEXR.OutputFile(filename, header)
    outfile.writePixels({channels[ci]: image[:, :, ci].astype(pixel_type_np(pixel_types[ci])).tostring() for ci in range(num_channels)})
    outfile.close()


def tonemap(image: np.ndarray, offset: float = 0.0, scale: float = 1.0, gamma: float = 1.0, as_uint8: bool = False,
            alpha: Union[np.ndarray, bool] = None, background: Union[np.ndarray, list, tuple] = (0., 0., 0.)):
    """apply simple scaling & gamma correction to HDR image; returns tonemapped 3D array clamped to [0, 1], optionally
    with alpha blending against a specified background, where alpha mask is either user specified as np.ndarray, or set
    to True when input image is in RGBA format; if background is set to None, the alpha channel will simply be stored
    as fourth channel in the tonemapped image again"""
    image = np.atleast_3d(image)

    if image.shape[2] == 4:
        # split alpha channel if available and requested
        if isinstance(alpha, bool) and alpha:
            alpha = image[:, :, 3:4]
            image = image[:, :, :3]

    # the actual tonemapping
    image = np.clip(scale * (image.astype(np.float32) - offset), 0., 1.) ** (1. / gamma)

    # alpha blending
    if alpha is not None:
        alpha = np.atleast_3d(alpha)
        if background is not None:
            background = np.array(background)
            while background.ndim < 3:
                # add fake spatial dimensions
                background = background[None]
            assert background.shape[2] == image.shape[2], 'background has %d channels but image has %d'\
                                                          % (background.shape[2], image.shape[2])
            image = alpha * image + (1 - alpha) * background
        else:
            # if background is disabled, don't blend, just write the alpha channel back (e.g. for PNG or WEBP export)
            image = np.concatenate((image, alpha), axis=2)

    # casting to uint8
    if as_uint8:
        image = (255 * image).astype(np.uint8)
    return image


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


def spec_image_to_srgb(image: np.ndarray,
                      wavelengths: Union[np.ndarray, List],
                      illuminant_name: str = 'E',
                      cmf_name: str = 'CIE 1931 2 Degree Standard Observer') -> np.ndarray:
    """convert spectral image to RGB given image as np.ndarray, wavelength sampling, illuminant & color matching
    functions (defaults to CIE 1931 RGB with equal energy (E) illuminant)"""
    import colour

    nc = image.shape[2]
    assert len(wavelengths) == nc
    assert illuminant_name in colour.SDS_ILLUMINANTS, 'illuminant must be one of ' + str(list(colour.SDS_ILLUMINANTS.keys()))
    assert cmf_name in colour.MSDS_CMFS, 'cmf must be one of ' + str(list(colour.MSDS_CMFS.keys()))

    spec_shape = colour.SpectralShape(wavelengths[0], wavelengths[-1], (wavelengths[-1] - wavelengths[0]) / (nc - 1))

    illuminant = deepcopy(colour.SDS_ILLUMINANTS[illuminant_name])
    illuminant = illuminant.align(shape=spec_shape)
    cmfs = deepcopy(colour.MSDS_CMFS[cmf_name])
    cmfs = cmfs.align(shape=spec_shape)
    image = colour.msds_to_XYZ(image, cmfs, illuminant, method='Integration', shape=spec_shape)
    image /= 100
    if cmf_name.lower().startswith('cie'):
        # convert CIE XYZ to sRGB
        image = colour.XYZ_to_sRGB(image,
                                   illuminant=colour.colorimetry.CCS_ILLUMINANTS[cmf_name][illuminant_name],
                                   apply_cctf_encoding=False)
    return image
