from copy import deepcopy
import numpy as np
from tqdm import tqdm

from pysmtb.utils import Dct, assign_masked, write_video
from test_rendering import test_rendering

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
