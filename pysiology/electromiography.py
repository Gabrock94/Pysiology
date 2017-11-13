import peakutils #peak detection
import numpy as np #to handle datas
import math #to handle mathematical stuff (example power of 2)
from scipy.signal import butter, lfilter, welch  #for signal filtering
import matplotlib.pyplot as plt


###############################################################################
#                                                                             #
#                              FUNCTIONS                                      #
#                                                                             #
###############################################################################
def getIEMG(rawEMGSignal):
    """ Sum of absolute values of EMG signal Amplitude
        IEMG = sum(|xi|) for i = 1 --> N
        Input: raw EMG Signal as list
        Output: integrated EMG    
    """
    
    IEMG = np.sum([abs(x) for x in rawEMGSignal])    
    return(IEMG)
    
def getMAV(rawEMGSignal):
    """ Average of EMG signal Amplitude
        IEMG = 1/N * sum(|xi|) for i = 1 --> N
        Input: raw EMG Signal as list
        Output: Mean Absolute Value    
    """
    
    MAV = 1/len(rawEMGSignal) *  np.sum([abs(x) for x in rawEMGSignal])    
    return(MAV)
    
def getMAV1(rawEMGSignal):
    """ Average of EMG signal Amplitude, modified 1
        IEMG = 1/N * sum(wi|xi|) for i = 1 --> N
        wi = {
              1 if 0.25N <= i <= 0.75N,
              0.5 otherwise
              }
        Input: raw EMG Signal as list
        Output: Mean Absolute Value    
    """
    wIndexMin = int(0.25 * len(rawEMGSignal))
    wIndexMax = int(0.75 * len(rawEMGSignal))
    absoluteSignal = [abs(x) for x in rawEMGSignal]
    IEMG = 0.5 * np.sum([x for x in rawEMGSignal[0:wIndexMin]]) + np.sum([x for x in rawEMGSignal[wIndexMin:wIndexMax]]) + 0.5 * np.sum([x for x in rawEMGSignal[wIndexMax:]])
    MAV1 = IEMG / len(rawEMGSignal)
    return(MAV1)
    
def getMAV2(rawEMGSignal):
    """ Average of EMG signal Amplitude, modified 2
        IEMG = 1/N * sum(wi|xi|) for i = 1 --> N
        wi = {
              1 if 0.25N <= i <= 0.75N,
              4i/N if i < 0.25N
              4(i-N)/N otherwise
              }
        Input: raw EMG Signal as list
        Output: Mean Absolute Value    
    """
    
    N = len(rawEMGSignal)
    wIndexMin = int(0.25 * N) #get the index at 0.25N
    wIndexMax = int(0.75 * N)#get the index at 0.75N

    temp = [] #create an empty list
    for i in range(0,wIndexMin): #case 1: i < 0.25N
        x = abs(rawEMGSignal[i] * (4*i/N))
        temp.append(x)
    for i in range(wIndexMin,wIndexMax+1): #case2: 0.25 <= i <= 0.75N
        x = abs(rawEMGSignal[i])
        temp.append(x)
    for i in range(wIndexMax+1,N): #case3; i > 0.75N
        x = abs(rawEMGSignal[i]) * (4*(i - N) / N)
        temp.append(x)
        
    MAV2 = np.sum(temp) / N
    return(MAV2)

def getSSI(rawEMGSignal):
    """ Summation of square values of the EMG signal
        SSI = sum(xi**2) for i = 1 --> N

        Input: raw EMG Signal as list
        Output: Simple Square Integral
    """
    
    SSI = np.sum([x**2 for x in rawEMGSignal])
    return(SSI)    
    
def getVAR(rawEMGSignal):
    """ Summation of average square values of the deviation of a variable
        VAR = (1 / (N - 1)) * sum(xi**2) for i = 1 --> N

        Input: raw EMG Signal as list
        Output: Simple Square Integral
    """
    
    SSI = np.sum([x**2 for x in rawEMGSignal])
    N = len(rawEMGSignal)
    VAR = SSI* (1 / (N - 1))
    return(VAR)    
    
def getTM(rawEMGSignal, order):
    """ 
        TM = (1 / N * sum(xi**order) for i = 1 --> N
        
        Input: raw EMG Signal as list
        Output: TM of order = order
    """
    N = len(rawEMGSignal)
    TM = abs((1/N) * np.sum([x**order for x in rawEMGSignal]))
    
    return(TM)    
    
def getRMS(rawEMGSignal):
    """ Get the root mean square of a signal
        RMS = (sqrt( (1 / N) * sum(xi**2))) for i = 1 --> N
        
        Input: raw EMG Signal as list
        Output: TM of order = order
    """
    N = len(rawEMGSignal)
    RMS = np.sqrt((1/N) * np.sum([x**2 for x in rawEMGSignal]))
    
    return(RMS)   
    
def getWL(rawEMGSignal):
    """ Get the waveform length of the signal
        WL = sum(|x(i+1) - xi|) for i = 1 --> N-1
        
        Input: raw EMG Signal as list
        Output: wavelength of the signal
    """
    N = len(rawEMGSignal)
    temp = []    
    for i in range(0,N-1):
        temp.append(abs(rawEMGSignal[i+1] - rawEMGSignal[i]))
    WL = sum(temp)
    return(WL)   
    
def getAAC(rawEMGSignal):
    """ Get the Average amplitude change
        AAC = 1/N * sum(|x(i+1) - xi|) for i = 1 --> N-1
        
        Input: raw EMG Signal as list
        Output: wavelength of the signal
    """
    N = len(rawEMGSignal)
    WL = getWL(rawEMGSignal)
    ACC = 1/N * WL
    return(ACC)
    
def getDASDV(rawEMGSignal):
    """ Get the standard deviation value of the the wavelength
        DASDV = sqrt( (1 / (N-1)) * sum((x[i+1] - x[i])**2 ) for i = 1 --> N - 1    
        
        Input: raw EMG Signal
        Output DASDV
    """
    
    N = len(rawEMGSignal)
    temp = []    
    for i in range(0,N-1):
        temp.append((rawEMGSignal[i+1] - rawEMGSignal[i])**2)
    DASDV = (1 / (N - 1)) * sum(temp)
    return(DASDV)
    
def getZC(rawEMGSignal, threshold):
    """ How many times does the signal crosses the 0 (+-threshold)
        ZC = sum([sgn(x[i] X x[i+1]) intersecated |x[i] - x[i+1]| >= threshold]) for i = 1 --> N - 1
        sign(x) = {
                    1, if x >= threshold
                    0, otherwise
                }
        Input:
            rawEMGSignal = EMG signal as list
            threshold = threshold to avoid fluctuations caused by noise and low voltage fluctuations
        Output:
            ZC index            
    """
    N = len(rawEMGSignal)
    positive = (rawEMGSignal[0] > threshold)
    ZC = 0
    for x in rawEMGSignal[1:]:
        if(positive):
            if(x < 0 -threshold):
                positive = False
                ZC += 1
        else:
            if(x > 0 + threshold):
                positive = True
                ZC += 1
    return(ZC)
    
    
    
def analyzeEMG(rawEMGSignal, samplerate,lowpass=50,highpass=20):
    """ This functions acts as entrypoint for the EMG Analysis.
        Input:
            rawEMGSignal = raw signal as list
            samplerate = samplerate of the signal
        Output:
    """ 
    resultsdict = {}
    
    #Preprocessing
    filteredEMGSignal = butter_lowpass_filter(rawEMGSignal, lowpass, samplerate, 2)#filter the signal with a cutoff at 1Hz and a 2th order Butterworth filter
    filteredEMGSignal = butter_highpass_filter(filteredEMGSignal, highpass, samplerate, 2)#filter the signal with a cutoff at 0.05Hz and a 2th order Butterworth filter
    
    #Time Domain Analysis
    resultsdict["IEMG"] = getIEMG(filteredEMGSignal)
    resultsdict["MAV"] = getMAV(filteredEMGSignal)
    resultsdict["MAV1"] = getMAV1(filteredEMGSignal)
    resultsdict["MAV2"] = getMAV2(filteredEMGSignal)
    resultsdict["SSI"] = getSSI(filteredEMGSignal)
    resultsdict["VAR"] = getVAR(filteredEMGSignal)
    resultsdict["TM3"] = getTM(filteredEMGSignal,3)
    resultsdict["TM4"] = getTM(filteredEMGSignal,4)
    resultsdict["TM5"] = getTM(filteredEMGSignal,5)
    resultsdict["RMS"] = getRMS(filteredEMGSignal)
    resultsdict["WL"] = getWL(filteredEMGSignal)
    resultsdict["AAC"] = getAAC(filteredEMGSignal)
    resultsdict["DASDV"] = getDASDV(filteredEMGSignal)
    resultsdict["ZC"] = getZC(filteredEMGSignal,0)
    
    return(filteredEMGSignal, resultsdict)
    
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
    import pprint
    fakesignal = []
    with open('/home/giulio/Scrivania/convertedEMG.pkl',"rb") as f:  # Python 3: open(..., 'rb')
        events = [30000]
        tmin = 0
        tmax = 5
        fakesignal = pickle.load(f) #load a fake signal
        samplerate = 1000
        for event in events:
            smin = tmin*samplerate + event
            smax = tmax*samplerate + event
            eventSignal = fakesignal[smin:smax]
            filtered, analyzedEMG = analyzeEMG(eventSignal,samplerate) #analyze it
            pprint.pprint(analyzedEMG)
#            plt.plot(filtered)
