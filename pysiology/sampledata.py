# -*- coding: utf-8 -*-
""" This module is used to load sample data from the share/data folder. """
import os
import pickle
import pkg_resources
###############################################################################
#                                                                             #
#                                  DEBUG                                      #
#                                                                             #
###############################################################################

def loadsampleECG():
    fakesignal = pkg_resources.resource_filename('pysiology','../share/data/convertedECG.pkl')
    with open(fakesignal,"rb") as f:  # Python 3: open(..., 'rb')
        return(pickle.load(f)) #load a fake signal
 
def loadsampleEMG():
    fakesignal = pkg_resources.resource_filename('pysiology','../share/data/convertedEMG.pkl')
    with open(fakesignal,"rb") as f:  # Python 3: open(..., 'rb')
        return(pickle.load(f)) #load a fake signal
    
def loadsampleEDA():
    fakesignal = pkg_resources.resource_filename('pysiology','../share/data/convertedEDA.pkl')
    with open(fakesignal,"rb") as f:  # Python 3: open(..., 'rb')
        return(pickle.load(f)) #load a fake signal