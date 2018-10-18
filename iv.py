# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 19:24:05 2018

@author: merzbach
"""
import numpy as np
from matplotlib import pyplot as plt

class iv:
    zoom_factor = 1.5
    x_zoom = True
    y_zoom = True
    x_stop_at_orig = True
    y_stop_at_orig = True
    
    def onclick(self, event):
        if event.dblclick:
            self.reset_zoom()
            self.mouse_down ^= event.button
        else:
            self.x_start = event.xdata
            self.y_start = event.ydata
            self.prev_delta_x = 0
            self.prev_delta_y = 0
            self.cur_xlims = self.ih.axes.axis()[0 : 2]
            self.cur_ylims = self.ih.axes.axis()[2 :]
            self.mouse_down |= event.button
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
            ('double' if event.dblclick else 'single', event.button,
            event.x, event.y, event.xdata, event.ydata))
            
    def onrelease(self, event):
        self.mouse_down ^= event.button
            
    def onmotion(self, event):
        if self.mouse_down == 1 and event.inaxes:
            delta_x = self.x_start - event.xdata
            delta_y = self.y_start - event.ydata
            print((delta_x, delta_y))
            self.ih.axes.axis((self.cur_xlims[0] + delta_x,
                               self.cur_xlims[1] + delta_x, 
                               self.cur_ylims[0] + delta_y,
                               self.cur_ylims[1] + delta_y))
            self.x_start += (delta_x - self.prev_delta_x)
            self.y_start += (delta_y - self.prev_delta_y)
            self.prev_delta_x = delta_x
            self.prev_delta_y = delta_y
            
    def onkeypress(self, event):
        print('key pressed: ', event.key)
        if event.key == 'a':
            min = np.min(self.image[:])
            max = np.max(self.image[:])
            self.offset = min
            self.scale = 1. / (max - min)
        elif event.key == 'G':
            self.gamma = 1.
        elif event.key == 'O':
            self.offset = 0.
        elif event.key == 'S':
            self.scale = 1.
        elif event.key == 'Z':
            # reset zoom
            self.ih.axes.autoscale(True)
        elif event.key == 'alt':
            self.alt = True
        elif event.key == 'control':
            self.control = True
        elif event.key == 'shift':
            self.shift = True
        elif event.key == 'left':
            self.imind = np.mod(self.imind - 1, self.nims)
        elif event.key == 'right':
            self.imind = np.mod(self.imind + 1, self.nims)
        else:
            return
        self.updateImage()
            
    def onkeyrelease(self, event):
        print('key released: ', event.key)
        if event.key == 'shift':
            self.shift = False
        elif event.key == 'shift':
            self.shift = False
        elif event.key == 'shift':
            self.shift = False
    
    def onscroll(self, event):
        if event.key == 'control':
            self.scale *= np.power(1.1, event.step)
            print('scale: %f', self.scale)
        elif event.key == 'shift':
            self.gamma *= np.power(1.1, event.step)
            print('gamma: %f', self.gamma)
        else:
            factor = np.power(self.zoom_factor, -event.step)
            self.zoom([event.xdata, event.ydata], factor)
            return
        self.updateImage()
        
    def zoom(self, pos, factor):
        lims = self.ih.axes.axis();
        xlim = lims[0 : 2]
        ylim = lims[2 : ]
        
        # compute interval lengths left, right, below and above cursor
        left = pos[0] - xlim[0];
        right = xlim[1] - pos[0];
        below = pos[1] - ylim[0];
        above = ylim[1] - pos[1];
        
        # zoom in or out
        if self.x_zoom:
            xlim = [pos[0] - factor * left, pos[0] + factor * right];
        if self.y_zoom:
            ylim = [pos[1] - factor * below, pos[1] + factor * above];
        
        # no zooming out beyond original zoom level
        if self.x_stop_at_orig:
            #xlim = [np.minimum(self.lims_orig[0], xlim[0]), np.maximum(self.lims_orig[1], xlim[1])];
            xlim = [np.maximum(self.lims_orig[0], xlim[0]), np.minimum(self.lims_orig[1], xlim[1])];
        
        if self.y_stop_at_orig:
            #ylim = [np.maximum(self.lims_orig[2], ylim[0]), np.minimum(self.lims_orig[3], ylim[1])];
            ylim = [np.minimum(self.lims_orig[2], ylim[0]), np.maximum(self.lims_orig[3], ylim[1])];
        
        # update axes
        if xlim[0] != xlim[1] and ylim[0] != ylim[1]:
            lims = (xlim[0], xlim[1], ylim[0], ylim[1])
            print('[%3.2f, %3.2f, %3.2f, %3.2f]' % self.lims_orig)
            print('[%3.2f, %3.2f, %3.2f, %3.2f]' % lims)
            self.ih.axes.axis(lims)
        return
        
    def reset_zoom(self):
        self.ih.axes.axis(self.lims_orig)
        
        
    def updateImage(self):
        self.ih.set_data(np.power(np.maximum(0., np.minimum(1., self.image[:, :, :, self.imind] * self.scale)), 1. / self.gamma))
    
    def __init__(self, image):
        # make input 4D (width x height x channels x images)
        while len(image.shape) < 4:
            image = np.reshape(image, image.shape + (1, ))
        # currently selected image
        self.w, self.h, self.nc = image.shape[0 : 3]
        self.image = image
        self.scale = 1.
        self.gamma = 1.
        self.offset = 0.
        self.imind = 0
        self.nims = image.shape[3]
        self.fig, self.ax = plt.subplots()
        self.ih = self.ax.imshow(np.zeros((self.w, self.h, self.nc)))
        self.updateImage()
        self.lims_orig = self.ih.axes.axis()
        self.mouse_down = 0
        self.x_start = 0
        self.y_start = 0
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid = self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        self.cid = self.fig.canvas.mpl_connect('motion_notify_event', self.onmotion)
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.onkeypress)
        self.cid = self.fig.canvas.mpl_connect('key_release_event', self.onkeyrelease)
        self.cid = self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        plt.show(block=True)