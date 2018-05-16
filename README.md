# PySiology
![GitHub release](https://img.shields.io/github/release/Gabrock94/Pysiology.svg)
[![PyPI version](https://badge.fury.io/py/pysiology.svg)](https://badge.fury.io/py/pysiology)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pysiology.svg)](https://pypi.python.org/pypi/pysiology/)
[![PyPI status](https://img.shields.io/pypi/status/pysiology.svg)](https://pypi.python.org/pypi/pysiology/)
[![Documentation Status](https://readthedocs.org/projects/pysiology/badge/?version=latest)](http://pysiology.readthedocs.io/en/latest/?badge=latest)


## Introduction
PySiology is a Python package used to analyze Physyological signals.
With pysiology you can easily analyze:
- Electromyographic signals
- Electrocardiographic signals
- Electrodermal activity signals

## Installation
PySiology can be installed using pip:
```bash
pip install pysiology
```
or downloading / cloning the repository and, from the root folder of the project, running:
```bash
python setup.py install
```


## Documentation
You can check the full documentation here: https://gabrock94.github.io/Pysiology/html/index.html

# Example
```python
import matplotlib.pyplot as plt #used for visualization purposes in this tutorial.

import pysiology
print(pysiology.__version__)


ECG = pysiology.sampledata.loadsampleECG() #load the sample ECG Signal
EMG = pysiology.sampledata.loadsampleEMG() #load the sample EMG Signal
GSR = pysiology.sampledata.loadsampleEDA() #load the sample GSR Signal

sr = 1000 #samplerate in Hz

#We can define the event in the way we prefer. 
#In this example I will use a 2 x nEvent matrix, containing the name of the event and the onset time.
events = [["A",10],
          ["B",20]]
eventLenght = 8 #lenght in seconds we want to use to compute feature estimation
results = {} #we will store the results in a dict for simplicity.
for event in events:
    startSample = sr * event[1] #samplerate of the signal multiplied by the onset of the event in s
    endSample = startSample + (sr * eventLenght) #Final sample to use for estimation
    results[event[0]] = {} #initialize the results
    results[event[0]]["ECG"] = pysiology.electrocardiography.analyzeECG(ECG[startSample:endSample],sr) #analyze the ECG signal
    results[event[0]]["EMG"] = pysiology.electromyography.analyzeEMG(EMG[startSample:endSample],sr) #analyze the EMG signal
    results[event[0]]["GSR"] = pysiology.electrodermalactivity.analyzeGSR(GSR[startSample:endSample],sr) #analyze the GSR signal

```

### Requirements
- Numpy
- Scipy
- Peakutils
- Matplotlib
   

## Contacts
Feel free to contact me for questions, suggestions or to give me advice as well at: giulio.gabrieli@studenti.unitn.it
