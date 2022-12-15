import imageio
import numpy as np
from pathlib import Path
import requests
from tqdm import tqdm

from pysmtb.btf import read_ubo_btf
from pysmtb.image import write_openexr, tonemap


def test_dfmf_loading():
    """simple test for UBO2014 BTF reading function"""
    url = 'https://cg.cs.uni-bonn.de/btf/UBO2014/leather/leather11_W400xH400_L151xV151.btf'
    urls = [
        'https://cg.cs.uni-bonn.de/btf/UBO2014/leather/leather04_resampled_W400xH400_L151xV151.btf',
        'https://cg.cs.uni-bonn.de/btf/UBO2014/leather/leather10_resampled_W400xH400_L151xV151.btf',
        'https://cg.cs.uni-bonn.de/btf/UBO2014/leather/leather11_resampled_W400xH400_L151xV151.btf',
    ]

    for url in tqdm(urls, 'downloading & reading BTF files'):
        filename = url.split('/')[-1].split('.')[0]
        material_name = filename.split('_')[0]
        filepath = Path('/tmp') / (filename + '.btf')

        if not filepath.exists():
            ret = requests.get(url, stream=True, verify=False)
            with open(filepath, 'wb') as fp:
                fp.write(ret.raw.data)

        meta, data = read_ubo_btf(filepath)

        # debug dump of height map
        if 'height_map' in data:
            write_openexr('/tmp/%s_heightmap.exr' % material_name, data.height_map, channels=['L'])
            imageio.imwrite('/tmp/%s_heightmap.png' % material_name, tonemap(data.height_map, normalize=True, normalization_percentiles=(1.0, 99.0), as_uint8=True))

        assert hasattr(data, 'U')
        assert hasattr(data, 'SxV')
        assert meta.width == 400 and meta.height == 400
        assert meta.num_lights == 151 and meta.num_views == 151

        # debug dump of top LV texture
        if isinstance(data.U, np.ndarray):
            def decode_texture(lind: int = 0, vind: int = 0) -> np.ndarray:
                ind = vind * meta.num_lights * 3 + lind * 3
                texture = data.SxV @ data.U[np.r_[ind:ind+3], :].T
                texture = texture.reshape(meta.height, meta.width, 3)

                drr_eps = 0.0
                if 'dynamic_range_reduction_method' in meta and meta.dynamic_range_reduction_method > 0:
                    drr_eps = meta.get('drr_eps', 1e-5)
                    drr_eps += meta.get('drr_offset', 0.0)

                return np.exp(texture) - drr_eps

            linds = [0, 20, 45, 50]
            vinds = [0, 0, 0, 0]

            for lind, vind in zip(linds, vinds):
                texture = decode_texture(lind=lind, vind=vind)
                write_openexr('/tmp/%s_texture_L=%03d_V=%03d.exr' % (material_name, lind, vind), texture, channels=['R', 'G', 'B'])
                imageio.imwrite('/tmp/%s_texture_L=%03d_V=%03d.png' % (material_name, lind, vind), tonemap(texture, normalize=True, normalization_percentiles=(1.0, 99.0), as_uint8=True))

        # test cropping
        meta, data = read_ubo_btf(filepath, roi=dict(left=0, top=0, width=50, height=50))
        assert meta.width == 50 and meta.height == 50

        assert data.U[0].shape[0] == 151 * 151
        assert data.SxV[0].shape[0] == 50 * 50

        # ensure projected view directions are within unit circle
        view_dirs_2d = np.array(meta.views_2d)
        assert np.all(-1 <= view_dirs_2d.min()) and np.all(view_dirs_2d <= 1)
        assert np.all(np.linalg.norm(view_dirs_2d, axis=1) <= 1)


if __name__ == '__main__':
    test_dfmf_loading()
