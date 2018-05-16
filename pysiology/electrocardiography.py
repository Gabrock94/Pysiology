import peakutils #peak detection for IBI / BPM
import numpy as np #to handle datas
import math #to handle mathematical stuff (example power of 2)
from scipy.signal import butter, lfilter  #for signal filtering
import scipy

def getIBI(peaks,samplerate):
    """ This function returns the IBI of a discrete heart signal.
    
        Input: peaks and samplerate of the ECG signal
        
        Output: IBI in ms
        
        :param peaks: list of peaks of the ECG signal
        :type peaks: list
        :param samplerate: samplerate of the signal in Hz
        :type samplerate: int
        :return: the mean IBI of the ECG signal
        :rtype: IBI (in ms) as float value
    """ 
    delta = []
    for peakIndex in range(1,len(peaks)):
        delta.append(((peaks[peakIndex] - peaks[peakIndex - 1]) / samplerate) * 1000)
    IBI = np.mean(delta) 
    return(IBI)
    
def getBPM(npeaks,nsample, samplerate):
    """ This function returns the BPM of a discrete heart signal.
    
        Input: number of peaks of the ECG signal,number of samples, samplerate of the signal
        
        Output: BPM
        
        :param npeak: number of peaks of the ECG signal
        :type npeak: int
        :param nsample: number of samples of the ECG signal
        :type nsample: int
        :param samplerate: samplerate of the signal in Hz
        :type samplerate: int
        :return: BPM of the ECG signal
        :rtype: float
    """ 
    samplelen = nsample / samplerate #lenght in seconds
    BPM = (npeaks * 60) / samplelen
    return(BPM)
    
def getSDNN(peaks,samplerate):
    """ This functions evaluate the standard deviation of intervals between heartbeats.
        It is often calculated over 24h period, or over short peridos of 5 mins.
        
        SDNN reflects all the cyclic components responsible for variability in the period of recording, therefore it represents total variability
        
        SDNN = sqrt((1/N-1) * sum(i=1 --> N)(rri - rrmean)^2
        
        Input: peaks of the ECG signal,samplerate of the signal
        
        Output: standard deviations of Intervals between heartbeats.
        
        :param peaks: list of peaks in the ECG signal
        :type peaks: list
        :param samplerate: samplerate of the signal in Hz
        :type samplerate: int
        :return: the SDNN of the ECG signal
        :rtype: float
    """
    delta = []
    for peakIndex in range(1,len(peaks)):
        delta.append(((peaks[peakIndex] - peaks[peakIndex - 1]) / samplerate) * 1000)
    SDNN = np.std(delta)
    return(SDNN)
    
def getSDSD(peaks,samplerate):
    """ This functions evaluate the the standard deviation of successive differences between adjacent R-R intervals.
    
        SDSD: sqrt((1 / (N - 1)) * sum(i=1 --> N)(RR i - mean(RR))**2)
        
        Input: peaks of the ECG signal,samplerate of the signal
        
        Output: the standard deviation of successive differences between adjacent R-R intervals
        
        :param peaks: list of peaks in the ECG signal
        :type peaks: list
        :param samplerate: samplerate of the signal in Hz
        :type samplerate: int
        :return: the SDSD of the ECG signal
        :rtype: float
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
    """ This functions evaluate the root mean square of successive differences between adjacent R-R intervals.
    
        RMSSD = sqrt((1 / (N - 1)) * sum(i=1 --> N)(RRdiff i - mean(RRdiff))**2)
        
        Input: peaks of the ECG signal,samplerate of the signal.
        
        Output: the root mean square of successive differences between adjacent R-R intervals.
        
        :param peaks: list of peaks in the ECG signal
        :type peaks: list
        :param samplerate: samplerate of the signal in Hz
        :type samplerate: int
        :return: the RMSSD of the ECG signal
        :rtype: float
        
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
    """ This functions evaluate pNN50, the proportion of differences greater than 50ms.
    
        Input: peaks of the ECG signal,samplerate of the signal
        
        Output: proportion of number of pairs of successive peaks that diffear by more than 50ms
        
        :param peaks: list of peaks in the ECG signal
        :type peaks: list
        :param samplerate: samplerate of the signal in Hz
        :type samplerate: int
        :return: the pNN50 of the ECG signal
        :rtype: float
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
    """ This functions evaluate pNN20, the proportion of differences greater than 20ms.
    
        Input: peaks of the ECG signal,samplerate of the signal
        
        Output: proportion of number of pairs of successive peaks that diffear by more than 20ms
        
        :param peaks: list of peaks in the ECG signal
        :type peaks: list
        :param samplerate: samplerate of the signal in Hz
        :type samplerate: int
        :return: the pNN20 of the ECG signal
        :rtype: float
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
    frequencies, psd = scipy.signal.periodogram(rawECGSignal, fs=samplerate, scaling="spectrum")
    return([psd,frequencies]) 
    
def getFrequencies(rawECGSignal, samplerate, llc=0.04, ulc=0.15, lhc=0.15,uhc=0.4, lvlc = 0.0033 , hvlc = 0.04 ):
    """This functions returns the power of low Frequencies, high frequencies and very low frequencies.
    
       Default Values have been taken from: Blood, J. D., Wu, J., Chaplin, T. M., Hommer, R., Vazquez, L., Rutherford, H. J., ... & Crowley, M. J. (2015). The variable heart: high frequency and very low frequency correlates of depressive symptoms in children and adolescents. Journal of affective disorders, 186, 119-126.
       
       It returns a dictionary with the amount of high, low and very low frequencies
       
       :param rawECGSignal: raw ECG signal
       :type rawECGSignal: list
       :param samplerate: samplerate of the ECG signal
       :type samplerate: int
       :param llc: lower cutoff of low frequencies
       :type llc: float
       :param ulc: upper cutoff of low frequencies
       :type ulc: float
       :param lhc: lower cutoff of high frequencies
       :type lhc: float
       :param uhc: high cutoff of high frequencies
       :type uhc: float
       :param lvlc: lower cutoff of very low frequencies
       :type lvlc: float
       :param uvlc: upper cutoff of very low frequencies
       :type uvlc: float

       :return: a dictionary containing the results of the frequency analysis
       :rtype: dictionary
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
    
#http://www.paulvangent.com/2016/03/15/analyzing-a-discrete-heart-rate-signal-using-python-part-1/
def analyzeECG(rawECGSignal,samplerate,preprocessing = True, highpass = 0.5, lowpass=2.5, min_dist = 500, ibi=True,bpm=True,sdnn = True,sdsd = True, rmssd = True,pnn50 = True, pnn20 = True, pnn50pnn20 = True, freqAnalysis = True, freqAnalysisFiltered = True):
    """ This is a simple entrypoint for ECG analysis. 
    
        You can use this function as model for your analysis or to extrapolate several features from an ECG signal.
        
        You can specify which features to evaluate or to exclude, as well as cutoff for frequencies and filter.
        
        :param data: ECG signal
        :type data: list
        :param samplerate: samplerate of the signal in Hz
        :type samplerate: int 
        :param preprocessing: whether to perform a simple preprocessing of the signal automatically
        :type preprocessing: boolean
        :param highpass: cutoff frequency for the high pass filter
        :type highpass: boolean
        :param lowpass: cutoff frequency for the low pass filter
        :type lowpass: boolean
        :param min_dist: minimum distance between peaks in ms. Used for peak detection
        :type min_dist: int
        :param ibi: whether or not to perform the IBI analysis
        :type ibi: boolean
        :param bpm: whether or not to perform the BPM analysis
        :type bpm: boolean
        :param sdnn: whether or not to perform the sdnn analysis
        :type sdnn: boolean
        :param sdsd: whether or not to perform the sdsd analysis
        :type sdsd: boolean
        :param rmssd: whether or not to perform the rmssd analysis
        :type rmssd: boolean
        :param pnn50: whether or not to perform the pNN50 analysis
        :type pnn50: boolean
        :param pnn20: whether or not to perform the pNN20 analysis
        :type pnn20: boolean
        :param pnn50pnn20: whether or not to perform the pNN50 on pNN20 ratio analysis
        :type pnn50pnn20: boolean
        :param freqAnalysis: whether or not to perform a frequency analysis analysis
        :type freqAnalysis: boolean
        :param freqAnalysisFiltered: whether or not to perform a frequency analysis automatically filtering the signal
        :type freqAnalysisFiltered: boolean

        :return: a dictionary containing the results of the ECG analysis 
        :rtype: list

    """
    #First we get the peaks
    if(preprocessing):
        filteredECGSignal = butter_lowpass_filter(rawECGSignal, lowpass, samplerate, 5)#filter the signal with a cutoff at 2.5Hz and a 5th order Butterworth filter
        filteredECGSignal = butter_highpass_filter(filteredECGSignal, highpass, samplerate, 5)#filter the signal with a cutoff at 2.5Hz and a 5th order Butterworth filter
    else:
        filteredECGSignal = rawECGSignal
    #min_dist = int(samplerate / 2) #Minimum distance between peaks is set to be 500ms
    min_dist = float(min_dist) / 1000 *samplerate
    peaks = peakutils.indexes(filteredECGSignal,min_dist=min_dist) #get the list of peaks
    resultsdict = {} #initialize the results dict.
    #for each analysis, check the boolean value and if true compute the results. Then append it to the final dict. 
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
        #We use a  try / except to prevent division by 0 or with null values
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
""" For debug purposes."""

if(__name__=='__main__'):
    import os
    import pickle
    import pprint
    import sampledata
    fakesignal = sampledata.loadsampleECG()
    events = [30000] #set a list of fake fake events
    tmin = 0 #start from the beginning of the events
    tmax = 8 #end from the beginning of the events
    samplerate = 1000 #samplerate of the fake signal
    for event in events: #for each event
        smin = tmin*samplerate + event
        smax = tmax*samplerate + event
        eventSignal = fakesignal[smin:smax]
        analyzedECG = analyzeECG(eventSignal,samplerate) #analyze it
        pprint.pprint(analyzedECG) #print the results of the analysis
