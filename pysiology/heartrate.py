import peakutils #peak detection for IBI / BPM
import numpy as np #to handle datas
import math #to handle mathematical stuff (example power of 2)
from scipy.signal import butter, lfilter  #for signal filtering
import scipy

def getIBI(peaks,samplerate):
    """ This function returns the IBI of a discrete heart signal
        Input: peaks of the ECG signal
        Output: IBI in ms
        
        :param peaks: list of peaks of the ECG signal
        :type peaks: list
        :rtype: IBI (in ms) as float value
    """ 
    delta = []
    for peakIndex in range(1,len(peaks)):
        delta.append(((peaks[peakIndex] - peaks[peakIndex - 1]) / samplerate) * 1000)
    IBI = np.mean(delta) 
    return(IBI)
    
def getBPM(npeaks,nsample, samplerate):
    """ This function returns the BPM of a discrete heart signal
        
        Input: number of peaks of the ECG signal,number of samples, samplerate of the signal
        Output: BPM
        
        :param npeak: number of peaks of the ECG signal
        :type npeak: int
        :param nsample: number of samples of the ECG signal
        :type nsample: int
        :param samplerate: samplerate of the signal in Hz
        :type samplerate: int
        
        :rtype: float
    """ 
    samplelen = nsample / samplerate #lenght in seconds
    BPM = (npeaks * 60) / samplelen
    return(BPM)
    
def getSDNN(peaks,samplerate):
    """ This functions evaluate the standard deviation of intervals between heartbeats
        SDNN = sqrt((1/N-1) * sum(i=1 --> N)(rri - rrmean)^2
        
        Input: peaks of the ECG signal,samplerate of the signal
        Output: standard deviations of Intervals between heartbeats
        
        :param peaks: list of peaks in the ECG signal
        :type peaks: list
        :param samplerate: samplerate of the signal in Hz
        :type samplerate: int
        
        :rtype: float
    """
    delta = []
    for peakIndex in range(1,len(peaks)):
        delta.append(((peaks[peakIndex] - peaks[peakIndex - 1]) / samplerate) * 1000)
    SDNN = np.std(delta)
    return(SDNN)
    
def getSDSD(peaks,samplerate):
    """ This functions evaluate the the standard deviation of successive differences between adjacent R-R intervals
        SDSD: sqrt((1 / (N - 1)) * sum(i=1 --> N)(RR i - mean(RR))**2)
        
        Input: peaks of the ECG signal,samplerate of the signal
        Output: the standard deviation of successive differences between adjacent R-R intervals:
    """
    delta = []
    for peakIndex in range(1,len(peaks)):
        delta.append(((peaks[peakIndex] - peaks[peakIndex - 1]) / samplerate) * 1000)
    
    differences = [] 
    for i in range(1,len(delta)):
        differences.append(delta[i] - delta[i-1])
        
    SDSD = np.std(differences)
    return(SDSD)
    
def getRMSSD(peaks,samplerate):
    """ This functions evaluate the root mean square of successive differences between adjacent R-R intervals
        RMSSD = sqrt((1 / (N - 1)) * sum(i=1 --> N)(RRdiff i - mean(RRdiff))**2)
        
        Input: peaks of the ECG signal,samplerate of the signal
        Output: the root mean square of successive differences between adjacent R-R intervals
    """
    delta = []
    for peakIndex in range(1,len(peaks)):
        delta.append(((peaks[peakIndex] - peaks[peakIndex - 1]) / samplerate) * 1000)
    
    differences = [] 
    for i in range(1,len(delta)):
        differences.append(math.pow(delta[i] - delta[i-1],2))
        
    RMSSD = np.std(differences)
    return(RMSSD)
    
def getPNN50(peaks,samplerate):
    """ This functions evaluate pNN20, the proportion of differences greater than 50ms.
        
        Input: peaks of the ECG signal,samplerate of the signal
        Output: the root mean square of successive differences between adjacent R-R intervals
    """
    delta = []
    for peakIndex in range(1,len(peaks)):
        delta.append(((peaks[peakIndex] - peaks[peakIndex - 1]) / samplerate) * 1000)
    
    differences = [] 
    for i in range(1,len(delta)):
        differences.append(delta[i] - delta[i-1])
        
    NN50 = [x for x in differences if x > 50]
    pNN50 = float(len(NN50)) / float(len(differences))
    return(pNN50)
    
def getPNN20(peaks,samplerate):
    """ This functions evaluate pNN20, the proportion of differences greater than 50ms.
        
        Input: peaks of the ECG signal,samplerate of the signal
        Output: the root mean square of successive differences between adjacent R-R intervals
    """
    delta = []
    for peakIndex in range(1,len(peaks)):
        delta.append(((peaks[peakIndex] - peaks[peakIndex - 1]) / samplerate) * 1000)
    
    differences = [] 
    for i in range(1,len(delta)):
        differences.append(delta[i] - delta[i-1])
        
    NN20 = [x for x in differences if x > 20]
    pNN20 = float(len(NN20)) / float(len(differences))
    return(pNN20)

def getPSD(rawECGSignal, samplerate):
#    frequencies, psd = scipy.welch(rawECGSignal, fs=samplerate,scaling="spectrum")
    frequencies, psd = scipy.signal.periodogram(rawECGSignal, fs=samplerate, scaling="spectrum")
    return([psd,frequencies]) 
    
def getFrequencies(rawECGSignal, samplerate, llc=0.04, ulc=0.15, lhc=0.15,uhc=0.4, lvlc = 0.0033 , hvlc = 0.04 ):
    """This functions returns the power of low Frequencies, high frequencies and very low frequencies.
       Default Values have been taken from: Blood, J. D., Wu, J., Chaplin, T. M., Hommer, R., Vazquez, L., Rutherford, H. J., ... & Crowley, M. J. (2015). The variable heart: high frequency and very low frequency correlates of depressive symptoms in children and adolescents. Journal of affective disorders, 186, 119-126.
       Input:
           rawECGSignal = raw ECG signal as list
           samplerate = sample rate in Hz
           llc, ulc = low frequency lower and upper cutoff
           lhc, uhc = high frequency lower and upper cutoff
           vllc, vluc = very low frequency lower and upper cutoff
      Output:
          PSD power of the low, high and very low frequencies
    """
    frequencyAnalysis = {}
    rawEMGPowerSpectrum, frequencies = getPSD(rawECGSignal,samplerate)
    frequencies = list(frequencies)
    #First we check for the closest value into the frequency list to the cutoff frequencies
    llc = min(frequencies, key=lambda x:abs(x-llc))
    ulc = min(frequencies, key=lambda x:abs(x-ulc))
    lhc = min(frequencies, key=lambda x:abs(x-lhc))
    uhc = min(frequencies, key=lambda x:abs(x-uhc))
    hvlc = min(frequencies, key=lambda x:abs(x-hvlc))
    lvlc = min(frequencies, key=lambda x:abs(x-lvlc))
    frequencyAnalysis["LF"] = sum([P for P in rawEMGPowerSpectrum[frequencies.index(llc):frequencies.index(ulc)]])
    frequencyAnalysis["HF"] = sum([P for P in rawEMGPowerSpectrum[frequencies.index(lhc):frequencies.index(uhc)]])
    frequencyAnalysis["VLF"] = sum([P for P in rawEMGPowerSpectrum[frequencies.index(lvlc):frequencies.index(hvlc)]])
    
    return(frequencyAnalysis)
    
#Define the filters
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
    
#http://www.paulvangent.com/2016/03/15/analyzing-a-discrete-heart-rate-signal-using-python-part-1/
def analyzeECG(rawECGSignal,samplerate,preprocessing = True, highpass = 0.5, lowpass=2.5, ibi=True,bpm=True,sdnn = True,sdsd = True, rmssd = True,pnn50 = True, pnn20 = True, pnn50pnn20 = True, freqAnalysis = True, freqAnalysisFiltered = True):
    """ This function analyze a discreate heart rate signal
        Input: 
            rawECGSignal = ecg signal as list
            samplerate = sample rate of the signal
            
        Output: BPM
    """ 
    #First we get the peaks
    if(preprocessing):
        filteredECGSignal = butter_lowpass_filter(rawECGSignal, lowpass, samplerate, 5)#filter the signal with a cutoff at 2.5Hz and a 5th order Butterworth filter
        filteredECGSignal = butter_highpass_filter(filteredECGSignal, highpass, samplerate, 5)#filter the signal with a cutoff at 2.5Hz and a 5th order Butterworth filter
    else:
        filteredECGSignal = rawECGSignal
    min_dist = int(samplerate / 2) #Minimum distance between peaks is set to be 500ms
    peaks = peakutils.indexes(filteredECGSignal,min_dist=min_dist)
    resultsdict = {}
    if(ibi):
        resultsdict["ibi"] =  getIBI(peaks,samplerate)
    if(bpm):
        resultsdict["bpm"] =  getBPM(len(peaks),len(rawECGSignal),samplerate)
    if(sdnn):
        resultsdict["sdnn"] =  getSDNN(peaks,samplerate)
    if(sdsd):
        resultsdict["sdsd"] =  getSDSD(peaks,samplerate)
    if(rmssd):
        resultsdict["rmssd"] =  getRMSSD(peaks,samplerate)
    if(pnn50):
        resultsdict["pnn50"] =  getPNN50(peaks,samplerate)
    if(pnn20):
        resultsdict["pnn20"] =  getPNN20(peaks,samplerate)
    if(pnn50pnn20):
        try:
            resultsdict["pnn50pnn20"] = resultsdict["pnn50"] / resultsdict["pnn20"]
        except:
            print("Unable to compute pnn50pnn20")
    if(freqAnalysis):
        resultsdict["frequencyAnalysis"] = getFrequencies(rawECGSignal,samplerate ) #unfiltered Signal
    if(freqAnalysisFiltered):
        resultsdict["frequencyAnalysisFiltered"] = getFrequencies(filteredECGSignal,samplerate ) #unfiltered Signal
    return(resultsdict)

###############################################################################
#                                                                             #
#                                  DEBUG                                      #
#                                                                             #
###############################################################################
""" For debug purposes"""

if(__name__=='__main__'):
    import os
    import pickle
    import pprint
    import matplotlib.pyplot as plt
    
    basepath = os.path.dirname(os.path.realpath(__file__)) #This get the basepath of the script
    datafolder = "/".join(basepath.split("/")[:-1])+"/data/"
    
    fakesignal = []
    with open(datafolder+"convertedECG.pkl","rb") as f:  # Python 3: open(..., 'rb')
        events = [30000] #set a list of fake fake events
        tmin = 0 #start from the beginning of the events
        tmax = 8 #end from the beginning of the events
        fakesignal = pickle.load(f) #load a fake signal
        samplerate = 1000 #samplerate of the fake signal
        for event in events: #for each event
            smin = tmin*samplerate + event
            smax = tmax*samplerate + event
            eventSignal = fakesignal[smin:smax]
            analyzedECG = analyzeECG(eventSignal,samplerate) #analyze it
            pprint.pprint(analyzedECG) #print the results of the analysis
