#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 21:12:15 2019

@author: spl
"""

import numpy as np

import PyQt5
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QCheckBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QSlider, QShortcut, QVBoxLayout, QWidget

class exposer(QMainWindow, QApplication):
    def __init__(self, target, **kwargs):
        QMainWindow.__init__(self, parent=None)
        
        self.target = target
        self.props = kwargs.get('props', None)
        
        self.widget = QWidget()
        self.form = QFormLayout()
        
        # set defaults for omitted parameters
        for name in self.props:
            if not 'type' in self.props[name]:
                self.props[name]['type'] = int
            if not 'style' in self.props[name]:
                self.props[name]['style'] = 'edit'
            if not 'limits' in self.props[name]:
                self.props[name]['limits'] = []
            if self.props[name]['style'] == 'slider' and not 'step' in self.props[name]:
                self.props[name]['step'] = 1
        
        # create controls
        self.controls = dict()
        for name in self.props:
            prop = self.props[name]
            if prop['style'] == 'edit':
                self.controls[name] = QLineEdit(str(getattr(self.target, name)))
                self.controls[name].setMinimumWidth(200)
                self.controls[name].editingFinished.connect(self.callback)
                
            elif prop['style'] == 'slider' or prop['style'] == 'sliderEdit':
                limits = prop['limits']
                step = prop['step']
                if len(limits) != 2:
                    raise Exception("slider limits must be specified as [min, max]!")
                
                self.controls[name] = QSlider(Qt.Horizontal)
                self.controls[name].setSingleStep(step)
                self.controls[name].setRange(0, (limits[1] - limits[0]) / step)
                self.controls[name].setValue(getattr(self.target, name) / step)
                self.controls[name].valueChanged.connect(self.callback)
                
                if prop['style'] == 'sliderEdit':
                    self.controls[name].sibling = QLineEdit(str(getattr(self.target, name)))
                    self.controls[name].sibling.setMinimumWidth(200)
                    self.controls[name].sibling.editingFinished.connect(self.callback)
                    self.controls[name].sibling.sibling = self.controls[name]
                    self.controls[name].sibling.tag = name
            
            self.controls[name].tag = name # store property name in control for access from the callback
            
            # add things into layout(s)
            if prop['style'] == 'sliderEdit':
                layout = QHBoxLayout()
                layout.addWidget(self.controls[name].sibling)
                layout.addWidget(self.controls[name])
                self.form.addRow(QLabel(name + ':'), layout)
                print(layout.stretch(0), layout.stretch(1))
            else:
                self.form.addRow(QLabel(name + ':'), self.controls[name])
        
        hbox = QHBoxLayout()
        hbox.addLayout(self.form)
        self.widget.setLayout(hbox)
        self.setCentralWidget(self.widget)
        self.show()
        
    def callback(self):
        try:
            target = self.sender()
            name = target.tag
            prop = self.props[name]
            if type(target) == PyQt5.QtWidgets.QSlider: #prop['style'] == 'slider':
                value = target.value() * float(prop['step']) + prop['limits'][0]
                
                if prop['style'] == 'sliderEdit':
                    # update edit box next to slider
                    self.controls[name].sibling.setText(str(value))
                
            elif type(target) == PyQt5.QtWidgets.QLineEdit: #prop['style'] == 'edit':
                value = target.text()
                value = prop['type'](value)
                limits = prop['limits']
                if not limits is None and len(limits):
                    if type(limits) == list and len(limits) == 2:
                        # clamp to valid range
                        value = max(limits[0], min(limits[1], value))
                    elif type(limits) == set:
                        # check if in set of valid values
                        if not value in limits:
                            return
                
                if prop['style'] == 'sliderEdit':
                    # update both slider and edit box with the potentially clamped value
                    self.controls[name].setValue((value - prop['limits'][0]) / prop['step'])
                    self.controls[name].sibling.setText(str(value))
                else:
                    # update edit box with potentially clamped value
                    self.controls[name].setText(str(value))
            else:
                print("else: ", type(target))
            # actual update of parameter in the target object
            setattr(self.target, name, value)
        except (RuntimeError, TypeError, NameError) as err:
            print("error: ", err)
        
class bla:
    def __init__(self):
        self.flup = 1
        self.flap = 1.
        self.flop = 3.
        self.flip = 1.
        
if __name__ == "__main__":
    b = bla()
    e = exposer(b, props=dict(flup={'type': int,   'style': 'edit',   'limits': [0, 10]},
                              flap={'type': float, 'style': 'edit',   'limits': [-np.inf, np.inf]},
                              flop={'type': float, 'style': 'slider', 'limits': [1, 10],    'step': 2},
                              flip={'type': float, 'style': 'sliderEdit', 'limits': [0.5, 2.5], 'step': 0.1}))