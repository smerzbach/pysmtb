#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 21:12:15 2019

@author: spl
"""

import numpy as np

import PyQt5
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton, QSlider, QSpinBox, QTextEdit, QWidget

class exposer(QWidget):
    def __init__(self, target, props, *args, **kwargs):
        super(exposer, self).__init__(*args, **kwargs)
        
        self.target = target
        self.props = props
        
        self.form = QFormLayout()
        
        # set defaults for omitted parameters
        for name in self.props:
            if not 'args' in self.props[name]:
                self.props[name]['args'] = dict()
                
            if not 'type' in self.props[name]:
                self.props[name]['type'] = int
            
            if not 'style' in self.props[name]:
                self.props[name]['style'] = 'edit'
            
            if not 'limits' in self.props[name]:
                self.props[name]['limits'] = []
            
            if (self.props[name]['style'] == 'slider' or self.props[name]['style'] == 'sliderEdit') and not 'step' in self.props[name]:
                self.props[name]['step'] = 1
            
            if (self.props[name]['style'] == 'slider' or self.props[name]['style'] == 'sliderEdit') and not 'sliderArgs' in self.props[name]:
                    self.props[name]['sliderArgs'] = self.props[name]['args']
            
            if self.props[name]['style'] == 'sliderEdit' and not 'editArgs' in self.props[name]:
                    self.props[name]['editArgs'] = self.props[name]['args']
        
        # create controls
        self.controls = dict()
        for name in self.props:
            prop = self.props[name]
            if prop['style'] == 'checkbox':
                self.controls[name] = QCheckBox(name, **prop['args'])
                self.controls[name].setChecked(bool(getattr(self.target, name)))
                self.controls[name].stateChanged.connect(self.callback)
                
            elif prop['style'] == 'combobox':
                self.controls[name] = QComboBox(**prop['args'])
                self.controls[name].addItems(prop['limits'])
                value = getattr(self.target, name)
                index = [i for i, it in enumerate(prop['limits']) if it == value]
                index = index[0]
                self.controls[name].setCurrentIndex(index)
                self.controls[name].currentIndexChanged.connect(self.callback)
            
            elif prop['style'] == 'edit':
                self.controls[name] = QLineEdit(str(getattr(self.target, name)), **prop['args'])
                self.controls[name].editingFinished.connect(self.callback)
                
            elif prop['style'] == 'slider' or prop['style'] == 'sliderEdit':
                limits = prop['limits']
                step = prop['step']
                
                if len(limits) != 2:
                    raise Exception("slider limits must be specified as [min, max]!")
                
                self.controls[name] = QSlider(Qt.Horizontal, None, **prop['sliderArgs'])
                self.controls[name].setSingleStep(step)
                self.controls[name].setRange(limits[0] / step, limits[1] / step)
                self.controls[name].setValue(getattr(self.target, name) / step)
                self.controls[name].valueChanged.connect(self.callback)
                
                if self.props[name]['style'] == 'sliderEdit':
                    self.controls[name].sibling = QLineEdit(str(getattr(self.target, name)), **prop['editArgs'])
                    self.controls[name].sibling.editingFinished.connect(self.callback)
                    self.controls[name].sibling.sibling = self.controls[name]
                    self.controls[name].sibling.tag = name
            
            elif prop['style'] == 'spinbox':
                self.controls[name] = QSpinBox(**prop['args'])
                self.controls[name].setRange(prop['limits'][0], prop['limits'][1])
                self.controls[name].setValue(prop['type'](getattr(self.target, name)))
                self.controls[name].valueChanged.connect(self.callback)
                
            self.controls[name].tag = name # store property name in control for access from the callback
            
            # add things into layout(s)
            if prop['style'] == 'sliderEdit':
                layout = QHBoxLayout()
                layout.addWidget(self.controls[name].sibling)
                layout.addWidget(self.controls[name])
                self.form.addRow(QLabel(name + ':'), layout)
                
            elif prop['style'] == 'checkbox':
                self.form.addRow(QLabel(''), self.controls[name])
                
            else:
                self.form.addRow(QLabel(name + ':'), self.controls[name])
        
        self.setLayout(self.form)
        self.show()
        
    def callback(self):
        try:
            target = self.sender()
            name = target.tag
            prop = self.props[name]
            #print(type(target))
            
            if isinstance(target, PyQt5.QtWidgets.QCheckBox):
                value = target.isChecked()
                
            elif isinstance(target, PyQt5.QtWidgets.QComboBox):
                value = target.currentText()
                
            elif isinstance(target, PyQt5.QtWidgets.QSlider):
                value = target.value() * float(prop['step']) # + prop['limits'][0]
                
                if prop['style'] == 'sliderEdit':
                    # update edit box next to slider
                    self.controls[name].sibling.setText(str(value))
                
            elif isinstance(target, PyQt5.QtWidgets.QLineEdit):
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
                    self.controls[name].setValue(value / prop['step']) #(value - prop['limits'][0]) / prop['step'])
                    self.controls[name].sibling.setText(str(value))
                else:
                    # update edit box with potentially clamped value
                    self.controls[name].setText(str(value))
            
            elif isinstance(target, PyQt5.QtWidgets.QSpinBox):
                value = target.value()
                value = prop['type'](value)
                
            else:
                print("else: ", type(target))
            # actual update of parameter in the target object
            setattr(self.target, name, value)
        except (RuntimeError, TypeError, NameError) as err:
            print("error: ", err)
        
class bla:
    def __init__(self):
        self.flup = 1
        self.fluup = 1
        self.flap = 1.
        self.flop = 3.
        self.flip = 1.
        self.flep = True
        self.flyp = 'asdf'
        
if __name__ == "__main__":
    b = bla()
    
    win = QMainWindow()
    win.setWindowTitle('foo')
    
    widget1 = QWidget()
    widget2 = QTextEdit()
    widget1.setMinimumSize(PyQt5.QtCore.QSize(200, 400))
    
    layout = QHBoxLayout()
    layout.addWidget(widget2, 1)
    layout.addWidget(widget1, 1)
    
    widget = QWidget()
    widget.setLayout(layout)
    win.setCentralWidget(widget)
    
    e = exposer(b, parent=widget1, props=dict(
                              flup={'type': int,   'style': 'edit',       'limits': [0, 10]},
                              fluup={'type': int,  'style': 'spinbox',    'limits': [0, 10]},
                              flap={'type': float, 'style': 'edit',       'limits': [-np.inf, np.inf]},
                              flop={'type': float, 'style': 'slider',     'limits': [1, 10],    'step': 2, 'args': {'tickInterval': 1, 'tickPosition': QSlider.TicksBelow}},
                              flip={'type': float, 'style': 'sliderEdit', 'limits': [0.5, 2.5], 'step': 0.1, 'sliderArgs': {'tickInterval': 10, 'tickPosition': QSlider.TicksBelow}},
                              flep={'type': bool,  'style': 'checkbox',   'limits': [False, True]},
                              flyp={'type': str,   'style': 'combobox',   'limits': ['a', 'b', 'c', 'asdf', 'd']}))
    
    win.show()    