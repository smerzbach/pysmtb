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
        elif event.inaxes:
            self.x_start = event.xdata
            self.y_start = event.ydata
            self.prev_delta_x = 0
            self.prev_delta_y = 0
            self.cur_xlims = self.ih.axes.axis()[0 : 2]
            self.cur_ylims = self.ih.axes.axis()[2 :]
            self.mouse_down |= event.button
            
    def onrelease(self, event):
        self.mouse_down ^= event.button
            
    def onmotion(self, event):
        if self.mouse_down == 1 and event.inaxes:
            delta_x = self.x_start - event.xdata
            delta_y = self.y_start - event.ydata
            self.ih.axes.axis((self.cur_xlims[0] + delta_x,
                               self.cur_xlims[1] + delta_x, 
                               self.cur_ylims[0] + delta_y,
                               self.cur_ylims[1] + delta_y))
            self.x_start += (delta_x - self.prev_delta_x)
            self.y_start += (delta_y - self.prev_delta_y)
            self.prev_delta_x = delta_x
            self.prev_delta_y = delta_y
    
    def onkeypress(self, event):
        if event.key == '?':
            self.print_usage()
        elif event.key == 'a':
            # trigger autoscale
            self.autoscale()
            return
        elif event.key == 'A':
            # toggle autoscale between user-selected percentiles or min-max
            self.autoscale_prctiles = not self.autoscale_prctiles
            self.autoscale()
            return
        elif event.key == 'c':
            # toggle on-change autoscale
            self.onchange_autoscale = not self.onchange_autoscale
            print('on-change autoscaling is %s' % ('on' if self.onchange_autoscale else 'off'))
        elif event.key == 'G':
            self.gamma = 1.
        elif event.key == 'L':
            # update axes for single image dimensions
            if self.is_collage:
                self.switch_to_single_image()
            else:
                # toggle showing collage
                self.is_collage = not self.is_collage
            # also disable per-image scaling limit computation
            self.per_image_scaling = not self.per_image_scaling
        elif event.key == 'O':
            self.offset = 0.
        elif event.key == 'p':
            self.per_image_scaling = not self.per_image_scaling
            print('per-image scaling is %s' % ('on' if self.per_image_scaling else 'off'))
            self.autoscale()
        elif event.key == 'S':
            self.scale = 1.
        elif event.key == 'Z':
            # reset zoom
            self.ih.axes.autoscale(True)
        elif event.key == 'alt':
            self.alt = True
            return
        elif event.key == 'control':
            self.control = True
            return
        elif event.key == 'shift':
            self.shift = True
            return
        elif event.key == 'left':
            self.switch_to_single_image()
            self.imind = np.mod(self.imind - 1, self.nims)
            print('image %d / %d' % (self.imind + 1, self.nims))
            if self.onchange_autoscale:
                self.autoscale()
                return
        elif event.key == 'right':
            self.switch_to_single_image()
            self.imind = np.mod(self.imind + 1, self.nims)
            print('image %d / %d' % (self.imind + 1, self.nims))
            if self.onchange_autoscale:
                self.autoscale()
                return
        else:
            return
        self.updateImage()
            
    def onkeyrelease(self, event):
        if event.key == 'shift':
            self.shift = False
        elif event.key == 'shift':
            self.shift = False
        elif event.key == 'shift':
            self.shift = False
    
    def onscroll(self, event):
        if event.key == 'ctrl+shift':
            # autoscale percentiles
            self.prctile *= np.power(1.1, event.step)
            self.prctile = np.minimum(100, self.prctile)
            print('auto percentiles: [%3.5f, %3.5f]' % (self.prctile, 100 - self.prctile))
            event.key = 'A'
            self.onkeypress(event)
        elif event.key == 'control':
            # scale
            self.scale *= np.power(1.1, event.step)
            print('scale: %f' % self.scale)
        elif event.key == 'shift':
            # gamma
            self.gamma *= np.power(1.1, event.step)
            print('gamma: %f' % self.gamma)
        elif event.inaxes:
            # zoom when inside image axes
            factor = np.power(self.zoom_factor, -event.step)
            self.zoom([event.xdata, event.ydata], factor)
            return
        else:
            # scroll through images when outside of axes
            self.switch_to_single_image()
            self.imind = int(np.mod(self.imind - event.step, self.nims))
            print('image %d / %d' % (self.imind + 1, self.nims))
            if self.onchange_autoscale:
                self.autoscale()
                return
        self.updateImage()
        
    
    def autoscale(self):
        # autoscale between user-selected percentiles
        if self.autoscale_prctiles:
            if self.per_image_scaling:
                min, max = np.percentile(self.image[:, :, :, self.imind], (self.prctile, 100 - self.prctile))
            else:
                min, max = np.percentile(self.image[:], (self.prctile, 100 - self.prctile))
        else:
            if self.per_image_scaling:
                min = np.min(self.image[:, :, :, self.imind])
                max = np.max(self.image[:, :, :, self.imind])
            else:
                min = np.min(self.image[:])
                max = np.max(self.image[:])
        self.offset = min
        self.scale = 1. / (max - min)
        print('scale: %3.5f, offset: %3.5f' % (self.scale, self.offset))
        self.updateImage()
        
    def collage(self):
        nc = int(np.ceil(np.sqrt(self.nims)))
        nr = int(np.ceil(self.nims / nc))
        # pad array so it matches the product nc * nr
        padding = nc * nr - self.nims
        coll = np.append(self.image, np.zeros((self.w, self.h, self.nc, padding)), axis=3)
        coll = np.reshape(coll, (self.w, self.h, self.nc, nc, nr))
        if self.border_width:
            # pad each patch by border if requested
            coll = np.append(coll, np.zeros((self.border_width, ) + coll.shape[1 : 5]), axis=0)
            coll = np.append(coll, np.zeros((coll.shape[0], self.border_width) + coll.shape[2 : 5]), axis=1)
        if self.transpose_collage:
            if self.transpose_frames:
                coll = np.transpose(coll, (4, 1, 3, 0, 2))
            else:
                coll = np.transpose(coll, (4, 0, 3, 1, 2))
        else:
            if self.transpose_frames:
                coll = np.transpose(coll, (3, 1, 4, 0, 2))
            else:
                coll = np.transpose(coll, (3, 0, 4, 1, 2))
        coll = np.reshape(coll, ((self.w + self.border_width) * nc, (self.h + self.border_width) * nr, self.nc))
        #self.ih.set_data(self.tonemap(coll))
        self.ih = self.ax.imshow(self.tonemap(coll))
        # todo: update original axis limits?
        #self.ih.axes.relim()
        #self.ih.axes.autoscale_view(True,True,True)        
    
    def switch_to_single_image(self):
        if self.is_collage:
            self.ih = self.ax.imshow(np.zeros((self.w, self.h, self.nc)))
        self.is_collage = False
        
    def reset_zoom(self):
        self.ih.axes.axis(self.lims_orig)
        
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
            self.ih.axes.axis(lims)
        return
        
    def tonemap(self, im):
        return np.power(np.maximum(0., np.minimum(1., (im - self.offset) * self.scale)), 1. / self.gamma)
        
    def updateImage(self):
        if self.is_collage:
            self.collage()
        else:
            self.ih.set_data(self.tonemap(self.image[:, :, :, self.imind]))
        
    def __init__(self, image, transposeFrames=False, transposeCollage=False):
        # make input 4D (width x height x channels x images)
        while len(image.shape) < 4:
            image = np.reshape(image, image.shape + (1, ))
        self.w, self.h, self.nc = image.shape[0 : 3]
        self.border_width = 0
        self.image = image
        self.scale = 1.
        self.gamma = 1.
        self.offset = 0.
        self.prctile = 0.1
        self.autoscale_prctiles = False
        self.onchange_autoscale = True
        self.per_image_scaling = True
        self.is_collage = False
        self.transpose_collage = transposeCollage
        self.transpose_frames = transposeFrames
        self.imind = 0 # currently selected image
        self.nims = image.shape[3]
        self.fig, self.ax = plt.subplots()
        self.ih = self.ax.imshow(np.zeros((self.w, self.h, self.nc)))
        plt.tight_layout()
        self.updateImage()
        if self.onchange_autoscale:
            self.autoscale()
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
        #plt.pause(10)
        
    def print_usage(self):
        print(' ')
        print('hotkeys: ')
        print('a: trigger autoscale')
        print('A: toggle autoscale of [min, max] or ')
        print('   [prctile_low, prctile_high] -> [0, 1], ')
        print('   prctiles can be changed via ctrl+shift+wheel')
        print('c: toggle autoscale on image change')
        print('G: reset gamma to 1')
        print('L: create collage by arranging all images in a ')
        print('   rectangular manner')
        print('O: reset offset to 0')
        print('p: toggle per image auto scale limit computations ')
        print('   (vs. globally over all images)')
        print('S: reset scale to 1')
        print('Z: reset zoom to 100%')
        print('left / right:         switch to next / previous image')
        print('page down / up:       go through images in ~10% steps')
        print('')
        print('wheel:                zoom in / out (inside image axes)')
        print('wheel:                switch to next / previous image')
        print('                      (outside image axes)')
        print('ctrl + wheel:         scale up / down')
        print('shift + wheel:        gamma up / down')
        print('ctrl + shift + wheel: increase / decrease autoscale')
        print('                      percentiles')
        print('left mouse dragged:   pan image')
        print('')