from visdom import Visdom

import numpy as np

class Plotter(object):
    """Plots to Visdom"""
    def __init__(self, envName='main', logFilename=None, port=8097):
        self.viz = Visdom(port=port, log_to_filename=logFilename, env=envName)
        self.env = envName
        self.windows = {}
        self.logFilename = logFilename

    def plot(self, windowName, legendName, x, y, ytype='linear', xlabel='', ylabel='', **kwargs):
        opts = {
                'legend': [legendName],
                'title': windowName,
                'xlabel': xlabel,
                'ylabel': ylabel,
                'ytype': ytype,
            }
        for key in kwargs:
            opts[key] = kwargs[key]
        if windowName not in self.windows:
            self.windows[windowName] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, name=legendName, opts=opts)
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.windows[windowName], name=legendName, update='append')
