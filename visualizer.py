from visdom import Visdom

import numpy as np
import os
import datetime

class Plotter(object):
    """Plots to Visdom"""
    def __init__(self, envName='main', logFilename=None, port=8097):
        self.viz = Visdom(port=port, log_to_filename=logFilename, env=envName)
        self.env = envName
        self.plots = {}
        self.logFilename = logFilename

    def plot(self, var_name, split_name, title_name, x, y, semilogy=False, xlabel='epochs'):
        opts = {
                'legend': [split_name],
                'title': title_name,
                'xlabel': xlabel,
                'ylabel': var_name
            }
        if semilogy:
            opts['layoutopts'] = {
                'plotly': {
                        'yaxis': {
                                'type': 'log',
                                'autorange': True,
                        }
                    }
                }
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=opts)
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
