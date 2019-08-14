#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 17:52:18 2019

@author: merzbach
"""

import logging
from datetime import datetime

class logger:
    def __init__(self, fname='debug.log', name='logger'):
        self.name = name
        self.fname = fname
        # create logger
        self.log = logging.getLogger(name=name)
        self.log.parent = None
        self.log.setLevel(logging.DEBUG)
        # create file handler
        self.fh = logging.FileHandler(fname)
        self.fh.setLevel(logging.DEBUG)
        # create console handler
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.DEBUG)
        # create formatter and add it to the handlers
        formatterFile = logging.Formatter(fmt='%(asctime)s_%(name)s_%(levelname)s: %(message)s', datefmt='%y%m%d_%H%M%S')
        self.fh.setFormatter(formatterFile)
        formatterConsole = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%y%m%d_%H%M%S')
        self.ch.setFormatter(formatterConsole)
        # add the handlers to the logger
        self.log.addHandler(self.fh)
        self.log.addHandler(self.ch)
        
    def print(self, *args):
        self.log.debug(*args)