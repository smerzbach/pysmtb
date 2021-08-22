from copy import deepcopy
import numpy as np
import os
from tqdm import tqdm

from pysmtb.utils import Dct, assign_masked, write_video
from test_rendering import test_rendering


# test OpenEXR I/O
from pysmtb.utils import read_openexr, write_openexr
image_float = np.random.rand(100, 100, 3)
image_half = image_float.astype(np.float16)
image_uint = ((2 ** 31 - 1) * image_float).astype(np.uint32)

os.makedirs('data', exist_ok=True)
write_openexr('data/test_img_half.exr', image_half, pixel_types='half')
write_openexr('data/test_img_float.exr', image_float, pixel_types='float')
write_openexr('data/test_img_uint.exr', image_uint, pixel_types='uint')

image_half_test = read_openexr('data/test_img_half.exr', pixel_type='half', sort_rgb=True)[0]
image_float_test = read_openexr('data/test_img_float.exr', pixel_type='float', sort_rgb=True)[0]
image_uint_test = read_openexr('data/test_img_uint.exr', pixel_type='uint', sort_rgb=True)[0]

test_half = np.testing.assert_almost_equal(image_half, image_half_test)
test_float = np.testing.assert_almost_equal(image_float, image_float_test)
test_uint = np.testing.assert_equal(image_uint, image_uint_test)

# simple test for rendering scripts
renderings, masks, buffers = test_rendering()

# export animation in different formats
lower, upper = np.percentile(np.array(renderings), (0.5, 99.5))
common_args = dict(frames=renderings, offset=lower, scale=1 / (upper - lower), gamma=1.4, framerate=5)
write_video('test.mp4', quality=1.0, **common_args)
write_video('test.gif', final_delay=100, **common_args)
# force usage of system ffmpeg to enable webp encoding
write_video('test.webp', quality=0.90, compression_level=4, ffmpeg_path='/usr/bin/', verbosity=100, **common_args)

# webp is the only option to export animations with transparent background
write_video('test_transparent.webp', quality=0.90, ffmpeg_path='/usr/bin/', masks=masks, **common_args)

# write simple HTML
text = 'blabla bla blaaab bla blala. ' * 100
html = f'''<html>
    <head>
        <style>
            .foreground {{
                background: url(test_transparent.webp) no-repeat;
                position: absolute;
                z-index: 10;
                width: 100%;
                height: 100%;
            }}
        </style>
    </head>
    <body>
        <div class="foreground">
        </div>
        {text}
    </body>
</html>
'''
with open('index.html', 'w') as fp:
    fp.write(html)
