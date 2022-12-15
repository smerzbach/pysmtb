import numpy as np
from pathlib import Path
from struct import unpack
from typing import Tuple, Union

from pysmtb.utils import Dct
from pysmtb.geometry import spherical_to_cartesian, cartesian_to_plane_stereographic


def read_ubo_btf(filepath: Union[str, Path], roi: dict = None) -> Tuple[Dct, Dct]:
    with open(filepath, 'rb') as fp:
        signature = fp.read(10).decode()

    if signature == '!DFMF08FCR':
        return read_ubo_btf_dfmf(filepath, roi=roi)
    elif signature == '!FMF12FCER':
        return read_ubo_fmf(filepath, roi=roi)
    else:
        raise Exception('unsupported BTF format: ' + str(signature))


def read_uint32(fp):
    return unpack('1I', fp.read(4))[0]


def read_common_header(fp, signature: str) -> Dct:
    pos_start = fp.tell()
    size = read_uint32(fp)
    version = read_uint32(fp)
    meta = Dct()
    meta.measurement_setup = fp.read(80)
    meta.image_sensor = fp.read(80)
    meta.light_source = fp.read(80)
    meta.ppmm = unpack('1f', fp.read(4))[0]
    meta.rgb_scale_factor = unpack('3f', fp.read(12))
    if version > 1:
        meta.cosine_in_data = read_uint32(fp)
    else:
        meta.cosine_in_data = False
    if version > 2:
        xml_size = read_uint32(fp)
        meta.xml = fp.read(xml_size)
    else:
        meta.xml = ''

    meta.num_channels = 3
    if version > 3:
        meta.num_channels = read_uint32(fp)
        meta.channel_names = []
        for ci in range(meta.num_channels):
            s = read_uint32(fp)
            meta.channel_names.append(fp.read(s))
    else:
        meta.channel_names = ['R', 'G', 'B']
    pos_end = fp.tell()

    if pos_start + size != pos_end:
        raise Exception('error parsing meta data from file ' + fp.name)

    # read directional sampling
    meta.num_views = read_uint32(fp)
    meta.num_lights_per_view = []
    meta.views_2d = []
    meta.views_3d = []
    meta.lights_2d = []
    for vi in range(meta.num_views):
        view_dir = unpack('2f', fp.read(8))
        view_dir = spherical_to_cartesian(*view_dir)
        meta.views_3d.append(view_dir)
        view_dir = cartesian_to_plane_stereographic(*view_dir)
        meta.views_2d.append(view_dir)
        num_lights = read_uint32(fp)
        light_dirs = np.fromfile(fp, dtype=np.float32, count=num_lights * 2).reshape(num_lights, 2)
        light_dirs = spherical_to_cartesian(light_dirs[:, 0], light_dirs[:, 1])
        light_dirs = np.array(cartesian_to_plane_stereographic(*light_dirs)).T
        meta.lights_2d.append(light_dirs)
        meta.num_lights_per_view.append(num_lights)
    meta.num_lights = np.max(meta.num_lights_per_view)

    meta.width = read_uint32(fp)
    meta.height = read_uint32(fp)

    if signature.endswith('R'):
        meta.num_rotations = read_uint32(fp)
        meta.rotations = np.fromfile(fp, dtype=np.float32, count=meta.num_rotations * 3).reshape(meta.num_rotations,                                                                3)
        meta.rotations = meta.rotations.reshape((meta.num_rotations, 3, 3))

    return meta


def read_ubo_fmf(filepath: Union[str, Path], roi: dict = None) -> Tuple[Dct, Dct]:
    with open(filepath, 'rb') as fp:
        signature = fp.read(10).decode()
        if signature != '!FMF12FCER':
            raise Exception('%s: BTF file is not a FMF file, expected signature !FMF12FCER, got %s' % (str(filepath), signature))

        data = Dct()
        meta = read_common_header(fp=fp, signature=signature)
        header_version = unpack('1B', fp.read(1))[0]

        if header_version >= 1:
            meta.dynamic_range_reduction_method = read_uint32(fp)

        if header_version >= 2:
            image_size = unpack('2I', fp.read(8))

            if np.prod(image_size) > 0:
                data.height_map = np.fromfile(fp, dtype=np.float16, count=np.prod(image_size)).reshape(image_size)

        if header_version >= 3:
            raise Exception('unsupported FMF format with header version ' + str(header_version))

        meta.num_components = read_uint32(fp)

        scalar_size = unpack('1B', fp.read(1))[0]
        if scalar_size == 2:
            dtype = np.float16
        elif scalar_size == 4:
            dtype = np.float32
        else:
            dtype = np.float64
        num_components = read_uint32(fp)
        num_rows = read_uint32(fp)
        num_cols = read_uint32(fp)
        data.S = np.fromfile(fp, dtype=dtype, count=num_components)
        data.U = np.fromfile(fp, dtype=dtype, count=num_rows * num_components).reshape(num_rows, num_components)
        data.SxV = np.fromfile(fp, dtype=dtype, count=num_cols * num_components).reshape(num_cols, num_components)

    return meta, data


def read_ubo_btf_dfmf(filepath: Union[str, Path], roi: dict = None):
    """read BTF (Bidirectional Texture Function) file in Uni Bonn DFMF (decorrelated full matrix factorization) format

    see also: https://cg.cs.uni-bonn.de/en/projects/btfdbb/download/ubo2014/

    optionally applies cropping (and optional downsampling) directly after loading, specify roi as dictionary with
    fields:
    left, top, width, height[, xstride, ystride]
    """

    with open(filepath, 'rb') as fp:
        signature = fp.read(10).decode()
        if signature != '!DFMF08FCR':
            raise Exception('BTF file %s has unknown header: %s' % (filepath, signature))

        meta = read_common_header(fp=fp, signature=signature)

        meta.num_components = read_uint32(fp)
        meta.color_model = unpack('1i', fp.read(4))[0]
        meta.color_mean = unpack('3f', fp.read(12))
        meta.color_transformation_matrix = np.fromfile(fp, dtype=np.float32, count=3 * 3).reshape(3, 3)

        data = Dct()
        data.S = []
        data.U = []
        data.SxV = []
        for ci in range(meta.num_channels):
            scalar_size = unpack('1B', fp.read(1))[0]
            if scalar_size == 2:
                dtype = np.float16
            elif scalar_size == 4:
                dtype = np.float32
            else:
                dtype = np.float64
            num_components = read_uint32(fp)
            num_rows = read_uint32(fp)
            num_cols = read_uint32(fp)
            data.S.append(np.fromfile(fp, dtype=dtype, count=num_components))
            data.U.append(np.fromfile(fp, dtype=dtype, count=num_rows * num_components).reshape(num_rows, num_components))
            data.SxV.append(np.fromfile(fp, dtype=dtype, count=num_cols * num_components).reshape(num_cols, num_components))

    if roi is not None:
        x0 = roi['left']
        x1 = roi['left'] + roi['width']
        xs = roi['xstride'] if 'xstride' in roi else 1
        y0 = roi['top']
        y1 = roi['top'] + roi['height']
        ys = roi['ystride'] if 'ystride' in roi else 1
        for ci in range(meta.num_channels):
            data.SxV[ci] = data.SxV[ci].reshape((meta.height, meta.width, -1))
            data.SxV[ci] = data.SxV[ci][y0: y1: ys, x0: x1: xs, :]
            height_new, width_new = data.SxV[ci].shape[:2]
            data.SxV[ci] = data.SxV[ci].reshape((height_new * width_new, -1))
        meta.height = height_new
        meta.width = width_new
    # hard-coded data not in the file
    meta.dynamic_range_eps = 1e-5
    return meta, data
