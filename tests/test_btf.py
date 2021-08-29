import numpy as np
import requests

from pysmtb.btf import read_ubo_btf_dfmf

def test_dfmf_loading():
    """simple test for UBO2014 BTF reading function"""
    filename = '/tmp/sample.btf'
    url = 'http://cg.cs.uni-bonn.de/fileadmin/btf/UBO2014/carpet/carpet05_W400xH400_L151xV151.btf'
    ret = requests.get(url, stream=True, verify=False)

    with open(filename, 'wb') as fp:
        fp.write(ret.raw.data)

    meta, data = read_ubo_btf_dfmf(filename)
    assert hasattr(data, 'U')
    assert hasattr(data, 'SxV')
    assert meta.width == 400 and meta.height == 400
    assert meta.num_lights == 151 and meta.num_views == 151

    # test cropping
    meta, data = read_ubo_btf_dfmf(filename, roi=dict(left=0, top=0, width=50, height=50))
    assert meta.width == 50 and meta.height == 50

    assert data.U[0].shape[0] == 151 * 151
    assert data.SxV[0].shape[0] == 50 * 50

    # ensure projected view directions are within unit circle
    view_dirs_2d = np.array(meta.views_2d)
    assert np.all(-1 <= view_dirs_2d.min()) and np.all(view_dirs_2d <= 1)
    assert np.all(np.linalg.norm(view_dirs_2d, axis=1) <= 1)

if __name__ == '__main__':
    test_dfmf_loading()
