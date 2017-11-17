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

def phasicGSRFilter(rawGSRSignal,samplerate):
    """ Apply a phasic filter to the signal, with +-4 seconds from each sample
        Input:
            rawGSRSignal = gsr signal as list
            samplerate = samplerate of the signal    
        Output:
            phasic filtered signal            
    """
    
    phasicSignal = []    
    for sample in range(0,len(rawGSRSignal)):
        smin = sample - 4 * samplerate #min sample index
        smax = sample + 4 * samplerate #max sample index
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
    """ Entry point for gsr analysis.
        Signal is filtered and downsampled, then a phasic filter is applied
        Input:
            rawGSRSignal = gsr signal as list            
            onset = onset for Peak Detection (uS)
            offset = offset for Peak Detection (uS)
        Output: 
            multi dimensional list, [onsetIndex,maxIndex,offsetIndex]*nPeaks
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
    
def GSRSCRFeaturesExtraction(filteredGSRSignal, samplerate, peak, presentationIndex=0):
    """ 
    This functions extract GSR SCR features: http://eda-explorer.media.mit.edu/static/SCR_withFeatures.png  
    Input: 
            rawGSRSignal: filtered GSR Signal as list
            samplerate: samplerate of the signak§
            peaks: list of peaks [peakStart, max, peakend]
            
        Ooutput:
            dict: {riseTime,Amplitude,EDAatApex,DecayTime (50%),SCRWidth (50%)}
            
    """  
    resultsDict = {}
    resultsDict["riseTime"] = (peak[1] - peak[0]) / samplerate
    resultsDict["latency"] = (peak[0] - presentationIndex) / samplerate
    resultsDict["amplitude"] = filteredGSRSignal[peak[1]] - filteredGSRSignal[peak[0]]
    resultsDict["halfAmplitude"] = float(resultsDict["amplitude"] / 2)
    resultsDict["halfAmplitudeIndex"] = filteredGSRSignal.index(min(filteredGSRSignal[peak[1]:peak[2]], key=lambda x:abs(x-resultsDict["halfAmplitude"])))
    resultsDict["halfAmplitudeIndexPre"] = filteredGSRSignal.index(min(filteredGSRSignal[peak[0]:peak[1]], key=lambda x:abs(x-resultsDict["halfAmplitude"])))
    resultsDict["EDAatApex"] = filteredGSRSignal[peak[1]]
    resultsDict["decayTime"] = (resultsDict["halfAmplitudeIndex"] - peak[1]) / samplerate
    resultsDict["SCRWitdth"] = (resultsDict["halfAmplitudeIndex"] - peak[0]) / samplerate
    return(resultsDict)
    
def analyzeGSR(rawGSRSignal,samplerate, lowpass=1,highpass=0.05):
    """ Entry point for gsr analysis.
        Signal is filtered and downsampled, then a phasic filter is applied
        Input:
            rawGSRSignal = gsr signal as list
            samplerate = samplerate of the signal                
            lowpass = cutoff for lowpass filter
            highpass = cutoff for highpass filter
        Output: 
            dictionary with all the results
    """
    resultsdict = {}    
    filteredGSRSignal = butter_lowpass_filter(rawGSRSignal, lowpass, samplerate, 2)#filter the signal with a cutoff at 1Hz and a 2th order Butterworth filter
    filteredGSRSignal = butter_highpass_filter(filteredGSRSignal, highpass, samplerate, 2)#filter the signal with a cutoff at 0.05Hz and a 2th order Butterworth filter
    scalingFactor = int(samplerate / 10) #scaling factor between the samplerate and 10Hz (downsampling factor)
    nsamples = int(len(filteredGSRSignal) / scalingFactor) #evalute the new number of samples for the downsampling
    filteredGSRSignal = scipy.signal.resample(filteredGSRSignal,nsamples) #downsample to 10Hz
    filteredGSRSignal = phasicGSRFilter(filteredGSRSignal,10) #apply a phasic filter
    peaks = findPeakOnsetAndOffset(filteredGSRSignal) #get peaks onset,offset and max
    nPeaks = len(peaks)
    return(filteredGSRSignal,peaks)

#Define the filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs #Nyquist frequeny is half the sampling frequency
    normal_cutoff = cutoff / nyq 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return(b, a)
    
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs #Nyquist frequeny is half the sampling frequency
    normal_cutoff = cutoff / nyq 
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return(b, a)
    
def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return(y)
    
def butter_highpass_filter(data, cutoff, fs, order):
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
    plt.clf()
    fakesignal = []
    with open(os.getcwd().replace('/pysiology/pysiology','/pysiology') + '/data/convertedEDA.pkl',"rb") as f:  # Python 3: open(..., 'rb')
        fakesignal = pickle.load(f) #load a fake signal
        analyzedGSR, peaks = analyzeGSR(fakesignal,1000) #analyze it
        
        
        plt.plot(analyzedGSR)
        for peak in peaks[0:1]:
            t0 = plt.scatter(peak[0],analyzedGSR[peak[0]],color='grey')
            t1 = plt.scatter(peak[1],analyzedGSR[peak[1]],color='green')
            t2 = plt.scatter(peak[2],analyzedGSR[peak[2]],color='grey')
            riseTime = plt.plot([peak[0],peak[1]],[0, 0],'--',color='blue',label="riseTime")
            amplitude = plt.plot([peak[1],peak[1]],[0, analyzedGSR[peak[1]]],'--',color='grey',label="Amplitude")
            results = GSRSCRFeaturesExtraction(analyzedGSR, 10, peak)
            plt.scatter(results["halfAmplitudeIndex"],analyzedGSR[results["halfAmplitudeIndex"]],color='purple')            
            SCRWidth = plt.plot([results["halfAmplitudeIndexPre"],results["halfAmplitudeIndex"]],[analyzedGSR[results["halfAmplitudeIndex"]],analyzedGSR[results["halfAmplitudeIndex"]]],'--',color='purple',label="width")
            decayTime = plt.plot([peak[1],results["halfAmplitudeIndex"]],[results["halfAmplitude"], results["halfAmplitude"]],'--',color='red', label="decayTime")
        for peak in peaks[1:]:
            t0 = plt.scatter(peak[0],analyzedGSR[peak[0]],color='grey')
            t1 = plt.scatter(peak[1],analyzedGSR[peak[1]],color='green')
            t2 = plt.scatter(peak[2],analyzedGSR[peak[2]],color='grey')
            riseTime = plt.plot([peak[0],peak[1]],[0, 0],'--',color='blue')
            amplitude = plt.plot([peak[1],peak[1]],[0, analyzedGSR[peak[1]]],'--',color='grey')
            results = GSRSCRFeaturesExtraction(analyzedGSR, 10, peak)
            plt.scatter(results["halfAmplitudeIndex"],analyzedGSR[results["halfAmplitudeIndex"]],color='purple')            
            SCRWidth = plt.plot([results["halfAmplitudeIndexPre"],results["halfAmplitudeIndex"]],[analyzedGSR[results["halfAmplitudeIndex"]],analyzedGSR[results["halfAmplitudeIndex"]]],'--',color='purple')
            decayTime = plt.plot([peak[1],results["halfAmplitudeIndex"]],[results["halfAmplitude"], results["halfAmplitude"]],'--',color='red')
        plt.legend() 