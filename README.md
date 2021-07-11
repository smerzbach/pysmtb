# pysmtb
python toolbox of (mostly) image-related helper / visualization functions 
```
from glob import glob

from pysmtb.iv import iv
from pysmtb.utils import read_exr

fns = glob('*.exr')
ims = [read_exr(fn)[0] for fn in fns]

iv(ims)
iv(ims, autoscale=False, scale=10, gamma=2)
iv(ims, collage=True)
```
![](examples/iv.jpg) ![](examples/iv_collage.jpg)

```
# add labels onto each image
iv(ims, labels=fns, annotate=True, annotate_numbers=False)
```
![](examples/iv_labels.jpg)

```
# test tight collage mode
ims1 = [np.random.rand(25, 15, 3) for _ in range(10)]
ims2 = [np.random.rand(10, 12, 3) for _ in range(10)]
ims3 = [np.random.rand(15, 12, 1) for _ in range(8)]
coll = collage(ims1 + ims2 + ims3, bw=1, tight=False)
coll_tight = collage(ims1 + ims2 + ims3, bw=1, tight=True)
iv.iv(dict(tight=coll_tight, non_tight=coll), collage=True, collageBorderWidth=1, collageBorderValue=1, annotate=True)
```
![](examples/collage_tight.png)
