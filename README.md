# PySiology
![GitHub release](https://img.shields.io/github/release/Gabrock94/Pysiology.svg)
[![PyPI](https://img.shields.io/pypi/v/pysiology.svg)](https://badge.fury.io/py/pysiology)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pysiology.svg)](https://pypi.python.org/pypi/pysiology/)
[![PyPI status](https://img.shields.io/pypi/status/pysiology.svg)](https://pypi.python.org/pypi/pysiology/)
[![PyPI downloads](https://img.shields.io/pypi/dm/pysiology.svg?label=PyPI%20downloads)](https://pypi.python.org/pypi/pysiology/)
[![Downloads](https://static.pepy.tech/badge/pysiology)](https://pepy.tech/project/pysiology)
[![Documentation Status](https://readthedocs.org/projects/pysiology/badge/?version=latest)](http://pysiology.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/109267664.svg)](https://zenodo.org/badge/latestdoi/109267664)



## Introduction
PySiology is a Python package used to analyze Physiological signals.
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

### Updating the package
To update the package via pip, you can use:
```bash
pip install --user --upgrade pysiology
```


## Documentation
You can check the full documentation here: https://pysiology.rtfd.io 

## WARNING
Sample data are not downloaded when using Pip. Please download the samples manually from the repository (https://github.com/Gabrock94/Pysiology/tree/master/share/data) and load them using 
```python
import pickle

with open("path/to/sample/data.pkl",'rb') as f:
          data = pickle.load(f)
```


## Example
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
## Cite
If you use PySiology, please cite:
> Gabrieli G., Azhari A., Esposito G. (2020) PySiology: A Python Package for Physiological Feature Extraction. In: Esposito A., Faundez-Zanuy M., Morabito F., Pasero E. (eds) Neural Approaches to Dynamics of Signal Exchanges. Smart Innovation, Systems and Technologies, vol 151. Springer, Singapore

## Requirements
- Numpy
- Scipy
- Peakutils
- Matplotlib

## Contacts
Feel free to contact me for questions, suggestions or to give me advice as well at: giulio.gabrieli@iit.it

## Scientific Publications that used pysiology
- Wiercinski, T., & Zawadzka, T. (2023). Short Paper: Late Fusion Approach for Multimodal Emotion Recognition Based on Convolutional and Graph Neural Networks.
- Gabrieli, G., Bornstein, M. H., Setoh, P., & Esposito, G. (2023). Machine learning estimation of users’ implicit and explicit aesthetic judgments of web-pages. Behaviour & Information Technology, 42(4), 392-402.- Hsu, S. M., Chen, S. H., &
- Momota, M. M. R., Morshed, B. I., Ferdous, T., & Fujiwara, T. (2023). Fabrication and Characterization of Inkjet Printed Flexible Dry ECG Electrodes. IEEE Sensors Journal, 23(7), 7917-7928.
- Warner, J., Gault, R., & McAllister, J. (2022, July). Optimised EMG pipeline for gesture classification. In 2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC) (pp. 3628-3631). IEEE.
- Momota, M. M. R., & Morshed, B. I. (2022). ML algorithms to estimate data reliability metric of ECG from inter-patient data for trustable AI-based cardiac monitors. Smart Health, 26, 100350.
- Chan, S. H. M., Qiu, L., Esposito, G., Mai, K. P., Tam, K. P., & Cui, J. (2021). Nature in virtual reality improves mood and reduces stress: evidence from young adults and senior citizens. Virtual reality, 1-16.
- Chan, S. H. M., Qiu, L., Esposito, G., & Mai, K. P. (2021). Vertical greenery buffers against stress: Evidence from psychophysiological responses in virtual reality. Landscape and Urban Planning, 213, 104127.
- Huang, T. R. (2021). Personal Resilience Can Be Well Estimated from Heart Rate Variability and Paralinguistic Features during Human–Robot Conversations. Sensors, 21(17), 5844.
- Aqajari, S. A. H., Naeini, E. K., Mehrabadi, M. A., Labbaf, S., Rahmani, A. M., & Dutt, N. (2020). Gsr analysis for stress: Development and validation of an open source tool for noisy naturalistic gsr data. arXiv preprint arXiv:2005.01834.
- Bizzego, A., Azhari, A., Campostrini, N., Truzzi, A., Ng, L. Y., Gabrieli, G., ... & Esposito, G. (2019). Strangers, friends, and lovers show different physiological synchrony in different emotional states. Behavioral Sciences, 10(1), 11.

## Coffee?
<a href='https://ko-fi.com/B0B3K45F' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://az743702.vo.msecnd.net/cdn/kofi2.png?v=0' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>
