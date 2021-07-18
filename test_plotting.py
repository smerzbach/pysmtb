import matplotlib.pyplot as plt
import numpy as np
from pysmtb.plotting import plot3, scatter3, quiver3, text3

plt.ion()

# generate some 3D points
xyz1 = np.random.multivariate_normal([0, 0, 0], np.array([[1, 0, 0], [0, 5, 0], [0, 0, 1]]), (100, ))
xyz2 = np.random.multivariate_normal([0, 0, 5], np.array([[10, 0, 0], [0, 1, 0], [0, 0, 1]]), (100, ))

# test 3D line plot
circle = np.stack((np.cos(np.linspace(0, 2 * np.pi, 50)),
                   np.sin(np.linspace(0, 2 * np.pi, 50)),
                   np.zeros((50,))), axis=0)
ph = plot3(circle, linestyle='-', marker='x')

# test 3D scatter plot
sh = scatter3(xyz1)
ax = sh.axes
sh2 = scatter3(xyz2, axes=ax)

# test 3D quiver plot
qh = quiver3(xyz2, xyz1, axis='auto')
ax = qh.axes
qh2 = quiver3(xyz1, -xyz2, axes=ax, axis='auto', color=[0.9, 0.6, 0.2])

# test 3D text
sh = scatter3(xyz1)
ax = sh.axes
text3(xyz1, ['%02d' % i for i in np.r_[:xyz1.shape[0]]], axes=ax)

print()