from setuptools import setup
import os

datadir = os.path.join("share","data")
datafiles = [(d,[os.path.join(d,f) for f in files]) for d, folders, files in os.walk(datadir)]

setup(name='pysiology',
    version='0.0.9.2',
    description='Physiological signal processing in Python',
    long_description="A simple python package for physiological signal processing (ECG,EMG,GSR). Tutorial and documentation can be found on the Github Repository or at pysiology.rtfd.io.",
    url='https://github.com/Gabrock94/Pysiology',
    download_url='https://github.com/Gabrock94/Pysiology/archive/0.0.7.tar.gz',
    author='Giulio Gabrieli',
    author_email='gack94@gmail.com',
    license='GPL-3.0',
    packages=['pysiology'],      
    install_requires=[
        'numpy',
        'peakutils',
        'scipy',
        'matplotlib'
    ],
    keywords = ["ECG","EMG","EDA","GSR","Physiology","Signal Processing"],
    classifiers = [ 
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License (GPL)',
        
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    zip_safe=False,
    include_package_data=True,
    data_files = datafiles)
