import peakutils #peak detection
import numpy as np #to handle datas
import math #to handle mathematical stuff (example power of 2)
from scipy.signal import butter, lfilter  #for signal filtering
import scipy
import matplotlib.pyplot as plt
###############################################################################
#                                                                             #
#                                Functions                                    #
#                                                                             #
###############################################################################

def phasicGSRFilter(rawGSRSignal,samplerate,seconds=4):
    """ Apply a phasic filter to the signal, with +- X seconds from each sample. Default is 10 seconds
    
        * Input:
            * rawGSRSignal = gsr signal as list
            * samplerate = samplerate of the signal    
            * seconds = number of seconds before and after each timepoint to use in order to compute the filtered value
        * Output:
            * phasic filtered signal            
        
        :param rawGSRSignal: raw GSR Signal
        :type rawGSRSignal: list
        :param samplerate: samplerate of the GSR signal in Hz
        :type samplerate: int
        :param seconds: seconds to use to apply the phasic filter
        :param seconds: int
        :return: filtered signal
        :rtype: list
    """
    
    phasicSignal = []    
    for sample in range(0,len(rawGSRSignal)):
        smin = sample - seconds * samplerate #min sample index
        smax = sample + seconds * samplerate #max sample index
        #is smin is < 0 or smax > signal length, fix it to the closest real sample
        if(smin < 0): 
            smin = sample
        if(smax > len(rawGSRSignal)):
            smax = sample
        #substract the mean of the segment
        newsample = rawGSRSignal[sample] - np.mean(rawGSRSignal[smin:smax])
        #move to th
        phasicSignal.append(newsample)
    return(phasicSignal)

def findPeakOnsetAndOffset(rawGSRSignal,onset=0.01,offset=0):
    """ This functions finds the peaks of a GSR signal
    
        * Input:
            * rawGSRSignal = GSR signal as list            
            * onset = onset for Peak Detection (uS)
            * offset = offset for Peak Detection (uS)
        * Output: 
            * multi dimensional list, \[onsetIndex,maxIndex,offsetIndex\] x nPeaks
            
        :param rawGSRSignal: GSR Signal to analyze
        :type rawGSRSignal: list
        :param onset: onset value for peak detection (in uS)
        :type onset: float
        :param offset: onset value for peak detection (in uS)
        :type offset: float
        :return: list of the peaks in the signal
        :rtype: float
    """
    
    listOfPeaks = [] #initialize the list of Peaks
    isOnset = False #set onset of False
    lastPeak = 0 #start lastpeak
    for point in range(0,len(rawGSRSignal)): #for each sample
        x = rawGSRSignal[point] #x is the value in uS of the sample
        if(isOnset): #if we are in onset phase
            if(x <= offset): #if x is below the offset
                peakOnset = max(rawGSRSignal[lastPeak:point])
                if(peakOnset >= onset):
                    listOfPeaks.append([lastPeak,rawGSRSignal.index(peakOnset),point]) #create the peak element
                isOnset = False #set isOnset to False
        else: #if we are in the offset phase
            if(x > offset): #if the point is above the onset
                lastPeak = point #memorize the onset index
                isOnset = True #switch onset to True
    return(listOfPeaks)
    
def GSRSCRFeaturesExtraction(filteredGSRSignal, samplerate, peak):
    """This functions extract GSR SCR features: http://eda-explorer.media.mit.edu/static/SCR_withFeatures.png  
    
        * Input: 
            * rawGSRSignal: filtered GSR Signal as list
            * samplerate: samplerate of the signak§
            * peak: list of peaks [peakStart, max, peakend]
                
        * Output:
            * dict: {riseTime,Amplitude,EDAatApex,DecayTime (50%),SCRWidth (50%)}
        
        :param rawGSRSignal: raw GSR Signal
        :type rawGSRSignal: list
        :param samplerate: samplerate of the GSR signal in Hz
        :type samplerate: int
        :param peak: a list containing the peak onset, max and offset indexes (as returned by the function findPeakOnsetAndOffset)
        :type peak: list
        :return: a dictionary with the results of the extracted features
        :rtype: dict
            
    """  
    
    resultsDict = {}
    try:
        resultsDict["peak"] = {"peakStart":peak[0],"peakMax":peak[1],"peakEnd":peak[2]}
        resultsDict["riseTime"] = (peak[1] - peak[0]) / samplerate
        resultsDict["latency"] = (peak[0]) / samplerate
        resultsDict["amplitude"] = filteredGSRSignal[peak[1]] - filteredGSRSignal[peak[0]]
        resultsDict["halfAmplitude"] = float(resultsDict["amplitude"] / 2)
        resultsDict["halfAmplitudeIndex"] = filteredGSRSignal.index(min(filteredGSRSignal[peak[1]:peak[2]], key=lambda x:abs(x-resultsDict["halfAmplitude"])))
        resultsDict["halfAmplitudeIndexPre"] = filteredGSRSignal.index(min(filteredGSRSignal[peak[0]:peak[1]], key=lambda x:abs(x-resultsDict["halfAmplitude"])))
        resultsDict["EDAatApex"] = filteredGSRSignal[peak[1]]
        resultsDict["decayTime"] = (resultsDict["halfAmplitudeIndex"] - peak[1]) / samplerate
        resultsDict["SCRWitdth"] = (resultsDict["halfAmplitudeIndex"] - peak[0]) / samplerate
    except:
        pass
    return(resultsDict)
    
def analyzeGSR(rawGSRSignal,samplerate, preprocessing=True, lowpass=1,highpass=0.05, phasic_seconds=10):
    """ Entry point for gsr analysis.
        Signal is filtered and downsampled, then a phasic filter is applied
        
        * Input:
            * rawGSRSignal = gsr signal as list
            * samplerate = samplerate of the signal
            * preprocessing = wether to perform or not a preprocessing of the signal                
            * lowpass = cutoff for lowpass filter
            * highpass = cutoff for highpass filter
        * Output: 
            * dictionary with all the results
        
        :param rawGSRSignal: raw GSR Signal
        :type rawGSRSignal: list
        :param samplerate: samplerate of the GSR signal in Hz
        :type samplerate: int
        :param preprocessing: wether to perform or not an automatic preprocessing of the signal
        :type preprocessing: true
        :param lowpass: cutoff frequency for the lowpass filter
        :type lowpass: float
        :param highpass: cutoff frequency for the highpass filter
        :type highpass: float
        :return: a dictionary with the results of the automatic GSR analysis
        :rtype: dict
    """
    resultsdict = {}    
    if(preprocessing):
        filteredGSRSignal = butter_lowpass_filter(rawGSRSignal, lowpass, samplerate, 2)#filter the signal with a cutoff at 1Hz and a 2th order Butterworth filter
        filteredGSRSignal = butter_highpass_filter(filteredGSRSignal, highpass, samplerate, 2)#filter the signal with a cutoff at 0.05Hz and a 2th order Butterworth filter
        scalingFactor = int(samplerate / 10) #scaling factor between the samplerate and 10Hz (downsampling factor)
        nsamples = int(len(filteredGSRSignal) / scalingFactor) #evalute the new number of samples for the downsampling
        filteredGSRSignal = scipy.signal.resample(filteredGSRSignal,nsamples) #downsample to 10Hz
        filteredGSRSignal = phasicGSRFilter(filteredGSRSignal,samplerate,seconds=phasic_seconds) #apply a phasic filter
    else:
        filteredGSRSignal = rawGSRSignal
    peaks = findPeakOnsetAndOffset(filteredGSRSignal) #get peaks onset,offset and max
    for peak in peaks:
        resultsdict[peaks.index(peak)] = GSRSCRFeaturesExtraction(filteredGSRSignal,10,peak)
    return(resultsdict)

#Define the filters
def butter_lowpass(cutoff, fs, order=5):
    """ This functions generates a lowpass butter filter
    
        :param cutoff: cutoff frequency
        :type cutoff: float
        :param cutoff: cutoff frequency
        :type cutoff: float
        :param fs: samplerate of the signal
        :type fs: float
        :param order: order of the Butter Filter
        :type order: int
        :return: butter lowpass filter
        :rtype: list
    """
    nyq = 0.5 * fs #Nyquist frequeny is half the sampling frequency
    normal_cutoff = cutoff / nyq 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return(b, a)
    
def butter_highpass(cutoff, fs, order=5):
    """ This functions generates a higpass butter filter
    
        :param cutoff: cutoff frequency
        :type cutoff: float
        :param cutoff: cutoff frequency
        :type cutoff: float
        :param fs: samplerate of the signal
        :type fs: float
        :param order: order of the Butter Filter
        :type order: int
        :return: butter highpass filter
        :rtype: list
    """
    nyq = 0.5 * fs #Nyquist frequeny is half the sampling frequency
    normal_cutoff = cutoff / nyq 
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return(b, a)
    
def butter_lowpass_filter(data, cutoff, fs, order):
    """ This functions apply a butter lowpass filter to a signal
    
        :param data: ECG signal
        :type data: list
        :param cutoff: cutoff frequency
        :type cutoff: float
        :param cutoff: cutoff frequency
        :type cutoff: float
        :param fs: samplerate of the signal
        :type fs: float
        :param order: order of the Butter Filter
        :type order: int
        :return: lowpass filtered ECG signal
        :rtype: list
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return(y)
    
def butter_highpass_filter(data, cutoff, fs, order):
    """ This functions apply a butter highpass filter to a signal
    
        :param data: ECG signal
        :type data: list
        :param cutoff: cutoff frequency
        :type cutoff: float
        :param cutoff: cutoff frequency
        :type cutoff: float
        :param fs: samplerate of the signal
        :type fs: float
        :param order: order of the Butter Filter
        :type order: int
        :return: highpass filtered ECG signal
        :rtype: list
    """
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return(y)

    
###############################################################################
#                                                                             #
#                                  DEBUG                                      #
#                                                                             #
###############################################################################
""" For debug purposes"""

if(__name__=='__main__'):
    import pickle
    import os
    import pprint
    import sampledata
    fakesignal =sampledata.loadsampleEDA() #load the sample GSR Signal
    GSRResults = analyzeGSR(fakesignal[1500:9500],1000,preprocessing=False,phasic_seconds=8) #analyze it
    pprint.pprint(GSRResults) #print the results for each peak found