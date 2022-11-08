#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:10:17 2022

this modules contains different utilities that cab be used for analysis. 

@author: giulio
"""

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import itertools

def leandriCC(signals, samplerate, windowsWidth, nAverages=None, tmin = 0, tmax = None):
    
    """ Implementation of the correlation analysis described in Campbell, J., & Leandri, M. (2020). 
    Using correlation analysis to assess the reliability of evoked potential components identified by signal averaging. 
    Journal of Neuroscience Methods, 340, 108752.
    
    :param signals: list of signals of shape [number of signals * lenght of signals]
    :type signals: list
    :param samplerate: sample rate of the signals]
    :type samplerate: int 
    :param windowsWidth: width of the windows used for analysis, in seconds
    :type windowsWidth: float
    :param nAverages: number of averages to use. It splits the signals into nAverages groups, and create an average for each group.
    :type nAverages: int
    :param tmin: start time for the analysis, in seconds. default is 0
    :type tmin: float    
    :param tmax: end time for the analysis, in seconds. if None it's automatically defined as the whole signal.
    :type tmax: float
    
    :return: a list containing times, mean correlation by window, and median correlation by window.
    :rtype: list

    """
    
    if(len(set([len(x) for x in signals])) != 1):
        raise Exception("All signals must have the same length") 
    
    #Get the length of the signals
    signalsLenght = len(signals[0])
    #get win width in sample
    windowsWidthSample = int(windowsWidth * samplerate)
    
    #initialize avg and median results
    avg = []
    median = []
    
    #checks
    if(nAverages != None):
        if(nAverages < 2):
            raise Exception("Number of averages should be >= 2 ") 
        else:
            signalspermean = int(len(signals) / nAverages)
            
            signals = [np.mean(signals[x:x+signalspermean], axis=0) for x in range(0, len(signals), signalspermean)]
        
    if(tmin != 0):
        if(tmin < 0):
            raise Exception("tmin should be >= 0") 
        else:
            tminSample = int(tmin * samplerate)
    else:
        tminSample = 0
        
    if(tmax != None):
        if(tmax < tmin):
            raise Exception("tmax should be >tmin") 
            
        elif(tmax > signalsLenght/samplerate):
            raise Exception("tmax should be < of signal length") 
            
        else:
            tmaxSample = int(tmax*samplerate)
            
    else:
        tmaxSample = signalsLenght
        tmax = signalsLenght/samplerate
        
    #loop for each window and pair of signals
    for w in range(tminSample, min(signalsLenght, tmaxSample), windowsWidthSample):
        thiswindow = []
        for pair in itertools.permutations(signals, 2):
           thiswindow.append(stats.pearsonr(pair[0][w:w+windowsWidthSample], pair[1][w:w+windowsWidthSample])[0])
        avg.append([np.mean(thiswindow)]*windowsWidthSample)
        median.append([np.median(thiswindow)]*windowsWidthSample)
    
    times = np.linspace(tmin, tmax, int((tmax-tmin) * samplerate))
    return([times, np.array(avg).flatten(), np.array(median).flatten()])


###############################################################################
#                                                                             #
#                                  DEBUG                                      #
#                                                                             #
###############################################################################
""" For debug purposes."""


if __name__ == "__main__":

    #Test for Debug
    start_time, end_time, sample_rate = [0,1,1000]
    time = np.arange(start_time, end_time, 1/sample_rate)
    frequency = 100
    amplitude = 1
    sinewave = amplitude * np.sin(2 * np.pi * frequency * time)
    
    testcases = 1000

    # Test
    signals = [np.random.normal(0,3,len(sinewave)) for i in range(testcases)]
    signals = [np.concatenate([signal[0:450], signal[450:550] + np.linspace(0,5, 100), signal[550:]]) for signal in signals]
   
    times , average, median = leandriCC(signals, sample_rate, 0.050, nAverages = 10, tmin = 0, tmax=0.9)

    # # create figure and axis objects with subplots()
    # fig,ax = plt.subplots()
    # # make a plot
    # ax.plot(np.linspace(start_time, end_time, end_time * sample_rate), np.mean(signals, axis=0), label='grand average', alpha=0.5)
    # # set x-axis label
    # ax.set_xlabel("time", fontsize = 14)
    # # set y-axis label
    # ax.set_ylabel("Signal", fontsize=14)
    
    # ax2=ax.twinx()
    # # make a plot with different y-axis using second axis object
    # ax2.plot(times, average, label='corr_average', color='red')
    # ax2.plot(times, median, label='corr_median', color='green')
    # ax2.set_ylabel("corr", fontsize=14)
    # plt.legend()
    # plt.show()