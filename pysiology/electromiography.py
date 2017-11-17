import peakutils #peak detection
import numpy as np #to handle datas
import math #to handle mathematical stuff (example power of 2)
from scipy.signal import butter, lfilter, welch  #for signal filtering
import matplotlib.pyplot as plt


###############################################################################
#                                                                             #
#                       TIME DOMAIN FEATURES                                  #
#                                                                             #
###############################################################################
""" Features have been taken from: 'Feature reduction and selection for EMG signal classification' """

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
    
def getMYOP(rawEMGSignal, threshold):
    """ The myopulse percentage rate (MYOP) is an average value of myopulse output.
        It is defined as one absolute value of the EMG signal exceed a pre-defined thershold value. 
        MYOP = (1/N) * sum(|f(xi)|) for i = 1 --> N
        f(x) = {
                1 if x >= threshold
                0 otherwise
        }
        Input:
            rawEMGSignal = EMG signal as list
            threshold = threshold to avoid fluctuations caused by noise and low voltage fluctuations
        Output:
            Myopulse percentage rate
    """
    N = len(rawEMGSignal)
    MYOP = len([1 for x in rawEMGSignal if abs(x) >= threshold]) / N
    return(MYOP)
    
def getWAMP(rawEMGSignal, threshold):
    """ Wilson or Willison amplitude is a measure of frequency information.
        It is a number of time resulting from difference between the EMG signal of two adjoining segments, that exceed a threshold
        WAMP = sum( f(|x[i] - x[i+1]|)) for n = 1 --> n-1
        f(x){
            1 if x >= threshold
            0 otherwise
        }
        Input:
            rawEMGSignal = EMG signal as list
            threshold = threshold to avoid fluctuations caused by noise and low voltage fluctuations
        Output:
            Wilson Amplitude value
    """
    N = len(rawEMGSignal)
    WAMP = 0
    for i in range(0,N-1):
        x = rawEMGSignal[i] - rawEMGSignal[i+1]
        if(x >= threshold):
            WAMP += 1
    return(WAMP)
    
def getSSC(rawEMGSignal,threshold):
    """ Number of times the slope of the EMG signal changes sign.
        SSC = sum(f( (x[i] - x[i-1]) X (x[i] - x[i+1]))) for i = 2 --> n-1
        f(x){
            1 if x >= threshold
            0 otherwise
        }
           
        Input: raw EMG Signal
        Output: number of Slope Changes
    """
    
    N = len(rawEMGSignal)
    SSC = 0
    for i in range(1,N-1):
        a,b,c = [rawEMGSignal[i-1],rawEMGSignal[i],rawEMGSignal[i+1]]
        if(a + b + c >= threshold *3 ): #computed only if the 3 values are above the threshold
            if(a < b > c or a > b < c ): #if there's change in the slope
                SSC += 1
    return(SSC)
    
def getMAVSLPk(rawEMGSignal, nseg):
    """ Mean Absolute value slope is a modified versions of MAV feature.
        The MAVs of adiacent segments are determinated. 
        MAVSLPk = MAV[k+1] - MAV[k]; k = 1,..,k+1
        
        Input: raw EMG signal as list
                nseg = number of segments to evaluate
                
        Output: 
             list of MAVs
    """
    N = len(rawEMGSignal)
    lenK = int(N / nseg) #length of each segment to compute
    MAVSLPk = []
    for s in range(0,N,lenK):
        MAVSLPk.append(getMAV(rawEMGSignal[s:s+lenK]))
    return(MAVSLPk)    


###############################################################################
#                                                                             #
#                       FREQUENCY DOMAIN FEATURES                             #
#                                                                             #
###############################################################################

def getMNF(rawEMGPowerSpectrum, frequencies):
    """ Obtain the mean frequency of the EMG signal, evaluated as the sum of 
        product of the EMG power spectrum and the frequency divided by total sum of the spectrum intensity
        MNF = sum(fPj) / sum(Pj) for j = 1 -> M 
        M = length of the frequency bin
        Pj = power at freqeuncy bin j
        fJ = frequency of the spectrum at frequency bin j
        Input: 
            rawEMGPowerSpectrum: PSD as list
            frequencies: frequencies of the PSD spectrum as list
        Output:
            Mean Frequency of the PSD
    """
    a = []
    for i in range(0, len(frequencies)):
        a.append(frequencies[i] * rawEMGPowerSpectrum[i])
    b = sum(rawEMGPowerSpectrum)
    MNF = sum(a) / b
    return(MNF)
    
def getMDF(rawEMGPowerSpectrum, frequencies):
    """ Obtain the Median Frequency of the PSD. 
        MDF is a frequency at which the spectrum is divided into two
        regions with equal amplitude, in other words, MDF is half of TTP feature
        
        Input: raw EMG Power Spectrum
                frequencies
        Output: Median Frequency  (Hz)
    """
    MDP = sum(rawEMGPowerSpectrum) * (1/2)
    for i in range(1, len(rawEMGPowerSpectrum)):
        if(sum(rawEMGPowerSpectrum[0:i]) >= MDP):
            return(frequencies[i])
            
def getPeakFrequency(rawEMGPowerSpectrum, frequencies):
    """ Obtain the frequency at which the maximum peak occur 
        Input:    
            raw EMG Power Spectrum as list
            frequencies as list
        Output:
            frequency in Hz
    """
    peakFrequency = frequencies[np.argmax(rawEMGPowerSpectrum)]
    return(peakFrequency)

def getMNP(rawEMGPowerSpectrum):
    """ This functions evaluate the mean power of the spectrum
        Mean Power = sum(Pj) / M, j = 1 --> M, M = len of the spectrum
        Input: EMG power spectrum
        Output: mean power
    """
    
    MNP = np.mean(rawEMGPowerSpectrum)
    return(MNP)
    
def getTTP(rawEMGPowerSpectrum):
    """ This functions evaluate the aggregate of the EMG power spectrum (aka Zero Spectral Moment)
        Input: raw EMG Power Spectrum
        Output: Total Power
    """
    
    TTP = sum(rawEMGPowerSpectrum)
    return(TTP)
        
def getSM(rawEMGPowerSpectrum, frequencies, order):
    """ Get the spectral moment of a spectrum
        SM = sum(fj*(Pj**order)), j = 1 --> M
        
        Input: raw EMG Power Spectrum
                frequencies as list
                order (int)
        Output: SM of order = order
    """
    SMo = []
    for j in range(0, len(frequencies)):
        SMo.append(frequencies[j]*(rawEMGPowerSpectrum[j] ** order))
    SMo = sum(SMo)
    return(SMo)   
    
def getFR(rawEMGPowerSpectrum, frequencies, llc=30, ulc=250, lhc=250,uhc=500):
    """ This functions evaluate the frequency ratio of the power spectrum. 
        Cut-off value can be decidec experimentally or from the MNF Feature See: Oskoei, M.A., Hu, H. (2006). GA-based feature subset selection for myoelectric classification.
        TODO: add automatic cutoff detection
        Input:
            raw EMG power spectrum as list,
            frequencies as list,
            llc = lower low cutoff
            ulc = upper low cutoff
            lhc = lower high cutoff
            uhc = upper high cutoff
        Output:
            Frequency Ratio
    """
    frequencies = list(frequencies)
    #First we check for the closest value into the frequency list to the cutoff frequencies
    llc = min(frequencies, key=lambda x:abs(x-llc))
    ulc = min(frequencies, key=lambda x:abs(x-ulc))
    lhc = min(frequencies, key=lambda x:abs(x-lhc))
    uhc = min(frequencies, key=lambda x:abs(x-uhc))
    
    LF = sum([P for P in rawEMGPowerSpectrum[frequencies.index(llc):frequencies.index(ulc)]])
    HF = sum([P for P in rawEMGPowerSpectrum[frequencies.index(lhc):frequencies.index(uhc)]])
    FR = LF / HF
    return(FR)

def getPSR(rawEMGPowerSpectrum,frequencies,n=20,fmin=10,fmax=500):
    """ Compute the Power Spectrum Ratio of the signal, defined as:
    Ratio between the energy P0 which is nearby the maximum va- lue of the EMG power spectrum and the energy P which is the whole energy of the EMG power spectrum
    Input:
        EMG power spectrum
        frequencies as list
        n = range around f0 to evaluate P0
        fmin = min frequency
        fmax = max frequency
    """
    
    frequencies = list(frequencies)
    
    #The maximum peak and frequencies are evaluate using the getPeakFrequency functions
    #First we check for the closest value into the frequency list to the cutoff frequencies
    peakFrequency = getPeakFrequency(rawEMGPowerSpectrum, frequencies)
    f0min = peakFrequency - n
    f0max = peakFrequency + n
    f0min = min(frequencies, key=lambda x:abs(x-f0min))
    f0max = min(frequencies, key=lambda x:abs(x-f0max))
    fmin = min(frequencies, key=lambda x:abs(x-fmin))
    fmax = min(frequencies, key=lambda x:abs(x-fmax))
    
    #here we evaluate P0 and P
    P0 = sum(rawEMGPowerSpectrum[frequencies.index(f0min):frequencies.index(f0max)])
    P = sum(rawEMGPowerSpectrum[frequencies.index(fmin):frequencies.index(fmax)])
    print(P0, P)
    PSR = P0 / P
    
    return(PSR)

def getVCF(SM0,SM1,SM2):
    """
        VCF = (1 / SM0)*sum(Pj*(fj - fc)**2),j = 1 --> M, = SM2 / SM0 - (SM1 /SM0) **2
    """
    VCF = (SM2 / SM0) - (SM1/SM0)**2
    return(VCF)
###############################################################################
#                                                                             #
#                           PREPROCESSING                                     #
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
    
def getPSD(rawEMGSignal, samplerate):
    frequencies, psd = welch(rawEMGSignal, fs=samplerate,
               window='hanning',   # apply a Hanning window before taking the DFT
               nperseg=256,        # compute periodograms of 256-long segments of x
               detrend='constant') # detrend x by subtracting the mean
    return([psd,frequencies])  
    
    
###############################################################################
#                                                                             #
#                              FILTERS                                        #
#                                                                             #
###############################################################################
    
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
#                               ENTRYPOINT                                    #
#                                                                             #
###############################################################################

    
def analyzeEMG(rawEMGSignal, samplerate,lowpass=50,highpass=20,threshold = 50,nseg=3,segoverlap=30):
    
    """ This functions acts as entrypoint for the EMG Analysis.
        Input:
            rawEMGSignal = raw signal as list
            samplerate = samplerate of the signal
            lowpass = lowpass cutoff in Hz
            highpass = highpass cutoff in Hz
            threshold for the evaluation of ZC,MYOP,WAMP,SSC
            nseg = number of segments for MAVSLPk, MHW,MTW
            segoverlap = Overlapping of the segments in percentage for MHW,MTW
        Output:
    """ 
    resultsdict = {"TimeDomain":{},"FrequencyDomain":{}}
    
    #Preprocessing
    filteredEMGSignal = butter_lowpass_filter(rawEMGSignal, lowpass, samplerate, 2)#filter the signal with a cutoff at 1Hz and a 2th order Butterworth filter
    filteredEMGSignal = butter_highpass_filter(filteredEMGSignal, highpass, samplerate, 2)#filter the signal with a cutoff at 0.05Hz and a 2th order Butterworth filter
    
    #Time Domain Analysis
    resultsdict["TimeDomain"]["IEMG"] = getIEMG(filteredEMGSignal)
    resultsdict["TimeDomain"]["MAV"] = getMAV(filteredEMGSignal)
    resultsdict["TimeDomain"]["MAV1"] = getMAV1(filteredEMGSignal)
    resultsdict["TimeDomain"]["MAV2"] = getMAV2(filteredEMGSignal)
    resultsdict["TimeDomain"]["SSI"] = getSSI(filteredEMGSignal)
    resultsdict["TimeDomain"]["VAR"] = getVAR(filteredEMGSignal)
    resultsdict["TimeDomain"]["TM3"] = getTM(filteredEMGSignal,3)
    resultsdict["TimeDomain"]["TM4"] = getTM(filteredEMGSignal,4)
    resultsdict["TimeDomain"]["TM5"] = getTM(filteredEMGSignal,5)
    resultsdict["TimeDomain"]["RMS"] = getRMS(filteredEMGSignal)
    resultsdict["TimeDomain"]["WL"] = getWL(filteredEMGSignal)
    resultsdict["TimeDomain"]["AAC"] = getAAC(filteredEMGSignal)
    resultsdict["TimeDomain"]["DASDV"] = getDASDV(filteredEMGSignal)
    resultsdict["TimeDomain"]["ZC"] = getZC(filteredEMGSignal,threshold)
    resultsdict["TimeDomain"]["MYOP"] = getMYOP(filteredEMGSignal,threshold)
    resultsdict["TimeDomain"]["WAMP"] = getWAMP(filteredEMGSignal,threshold)
    resultsdict["TimeDomain"]["SSC"] = getSSC(filteredEMGSignal,threshold)
    resultsdict["TimeDomain"]["MAVSLPk"] = getMAVSLPk(filteredEMGSignal,nseg)
    
    #Frequency Domain Analysis
    rawEMGPowerSpectrum, frequencies = getPSD(filteredEMGSignal,samplerate)
    
    resultsdict["FrequencyDomain"]["MNF"] = getMNF(rawEMGPowerSpectrum, frequencies)
    resultsdict["FrequencyDomain"]["MDF"] = getMDF(rawEMGPowerSpectrum, frequencies)
    resultsdict["FrequencyDomain"]["PeakFrequency"] = getPeakFrequency(rawEMGPowerSpectrum, frequencies)
    resultsdict["FrequencyDomain"]["MNP"] = getMNP(rawEMGPowerSpectrum)
    resultsdict["FrequencyDomain"]["TTP"] = getTTP(rawEMGPowerSpectrum)
    resultsdict["FrequencyDomain"]["SM1"] = getSM(rawEMGPowerSpectrum,frequencies,1)
    resultsdict["FrequencyDomain"]["SM2"] = getSM(rawEMGPowerSpectrum,frequencies,2)
    resultsdict["FrequencyDomain"]["SM3"] = getSM(rawEMGPowerSpectrum,frequencies,3)
    resultsdict["FrequencyDomain"]["FR"] = getFR(rawEMGPowerSpectrum,frequencies)
    resultsdict["FrequencyDomain"]["PSR"] = getPSR(rawEMGPowerSpectrum,frequencies)
    resultsdict["FrequencyDomain"]["VCF"] = getVCF(resultsdict["FrequencyDomain"]["TTP"],resultsdict["FrequencyDomain"]["SM1"],resultsdict["FrequencyDomain"]["SM2"])
    
    return(resultsdict)
    
    
###############################################################################
#                                                                             #
#                                  DEBUG                                      #
#                                                                             #
###############################################################################
""" For debug purposes. This runs only if this file is loaded directly and not imported """

if(__name__=='__main__'):
    import pickle
    import pprint
    fakesignal = []
    with open('/home/giulio/Scrivania/convertedEMG.pkl',"rb") as f:  # Python 3: open(..., 'rb')
        events = [30000] #set a fake event
        tmin = 0 #start from the beginning of the events
        tmax = 5 #end from the beginning of the events
        fakesignal = pickle.load(f) #load a fake signal
        samplerate = 1000 #samplerate of the fake signal
        for event in events: #for each event
            smin = tmin*samplerate + event
            smax = tmax*samplerate + event
            eventSignal = fakesignal[smin:smax]
            analyzedEMG = analyzeEMG(eventSignal,samplerate) #analyze it
            pprint.pprint(analyzedEMG) #print the results of the analysis
