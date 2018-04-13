# Pysiology
## Introduction
Pysiology is a Python package used to analyze Physyological signals.
With pysiology you can easily analyze:
- Electromyography signals
- Electrocardiography signals
- Electrodermal activity signals

## Installation
pip install pysiology

## Documentation
You can check the full documentation here: https://gabrock94.github.io/Pysiology/html/index.html

# Example
```python
import os #used for reading files and directories
import pickle #used to open pickle files.
import pysiology


datafolder = "path/to/data/" #path to the data folder
fileECG = "ECGSignal.pkl" #filename of the ECG signal
fileEDA = "EDASignal.pkl" #filename of the EDA signal
fileEMG = "EMGSignal.pkl" #filename of the EMG signal

if(__name__ == "__main__"):
    samplerate = 1000 #samplerate of the signals
    eventDuration = 8 #event duration in seconds 
    
    #First we load our data
    with open(datafolder+fileECG,"rb") as f:
        rawECGSignal = pickle.load(f)
    with open(datafolder+fileEDA,"rb") as f:
        rawEDASignal = pickle.load(f)
    with open(datafolder+fileEMG,"rb") as f:
        rawEMGSignal = pickle.load(f)
        
    #Here we create some fake events
    events = [["A",30],["B",60],["C",90]] #id, starttime in seconds
    results = {}
    for event in events:
        startSample = event[1] * samplerate #First sample of the event
        endSample = eventDuration*samplerate + startSample #Final sample of the event
        results[event[0]] = {} #create a dict for this event results
        results[event[0]]["ECG"] = pysiology.heartrate.analyzeECG(rawECGSignal[startSample:endSample],samplerate) #analyze the ECG signal
        results[event[0]]["EDA"] = pysiology.electrodermalactivity.analyzeGSR(rawEDASignal[startSample:endSample],samplerate) #analyze the GSR signal
        results[event[0]]["EMG"] = pysiology.electromiography.analyzeEMG(rawEMGSignal[startSample:endSample],samplerate) #analyze the EMG signal

```

## Installation
### Requirements
- Numpy
- Scipy
- peakutils

# Credits
   

## Contacts
Feel free to contact me for questions, suggestions or to give me advice as well at: giulio.gabrieli@studenti.unitn.it
